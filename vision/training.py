"""Train separate RGB or depth gesture models from prepared video samples.

The dataset decodes frames lazily; this module never writes source videos,
depth images or annotations. Every output is confined to runs/rgb or runs/depth
in this repository. Complete source trials must already be partitioned by the
shared protocol code before this trainer receives them.

AdamW minimizes cross-entropy. Validation macro F1 selects the best checkpoint,
with validation loss breaking ties; ReduceLROnPlateau lowers the learning rate
when validation loss stalls. The held-out test split is evaluated exactly once,
after restoring the selected checkpoint. Without validation, the last epoch is
selected and validation-dependent scheduling and early stopping are disabled.

Each fold saves split provenance, checkpoints, epoch history, runtime details,
per-gesture test metrics, predictions and a confusion matrix. The final test
report is also printed directly in the terminal. Checkpoints contain only
tensors and ordinary Python values and can be loaded with weights_only=True.
"""

import csv
from importlib import metadata as package_metadata
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.artifacts import PROJECT_ROOT, write_json
from scripts.labels import validate_label_mapping
from scripts.metrics import classification_metrics, format_test_report


def _output_directory(output_dir, modality=None):
    root = Path(PROJECT_ROOT).resolve()
    runs = root / "runs"
    output = Path(output_dir).resolve()
    if not runs.resolve().is_relative_to(root):
        raise ValueError("The runs directory must not redirect outside this repository")
    try:
        parts = output.relative_to(runs).parts
    except ValueError as error:
        raise ValueError("Video artifacts must stay under runs/rgb or runs/depth") from error
    if len(parts) < 2 or parts[0] not in ("rgb", "depth"):
        raise ValueError("Video artifacts need a run directory under runs/rgb or runs/depth")
    if modality is not None and parts[0] != modality:
        raise ValueError("Output directory does not match the configured modality")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _artifact(output, filename):
    path = output / filename
    if path.resolve().parent != output:
        raise ValueError("An output file must not redirect outside the run directory")
    return path


def _write_json(output, filename, value):
    write_json(_artifact(output, filename), value)


def save_prepared_fold(fold, output_dir):
    """Save split metadata, retaining original source IDs and relative paths."""
    output = _output_directory(output_dir)
    for part in ("train", "validation", "test"):
        _write_json(output, part + "_samples.json", fold["splits"][part])
    _write_json(output, "split_summary.json", fold.get("split_summary", {}))
    _write_json(output, "protocol.json", fold.get("protocol_metadata", {}))
    _write_json(output, "normalization.json", fold.get("normalization", {}))
    _write_json(output, "excluded_records.json", fold.get("excluded_records", []))


def choose_device(value="auto"):
    """Resolve the requested device and give an explicit CUDA availability error."""
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use auto, cpu, cuda, or a CUDA device such as cuda:0")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; choose cpu or install a CUDA PyTorch build")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index is unavailable")
    return device


def set_seed(seed, deterministic=True):
    if type(seed) is not int or seed < 0 or type(deterministic) is not bool:
        raise ValueError("seed must be a nonnegative integer and deterministic a boolean")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def environment_info(device, amp=False):
    packages = {}
    for name in ("opencv-python", "opencv-python-headless", "Pillow"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            pass
    return dict(python=sys.version.split()[0], python_executable=sys.executable,
                torch=str(torch.__version__), numpy=np.__version__, packages=packages,
                device=str(device), amp=amp, cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cudnn_version=torch.backends.cudnn.version(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def _positive_number(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(name + (" must be finite and nonnegative" if allow_zero else " must be finite and positive"))
    return value


def _settings(config):
    raw = config.get("training") or {}
    if not isinstance(raw, dict):
        raise ValueError("training must be a configuration object")
    s = dict(epochs=30, batch_size=4, learning_rate=0.0001, weight_decay=0.001,
             seed=42, device="auto", cpu_threads=1, deterministic=True, num_workers=0,
             amp=True, early_stopping_patience=10, gradient_clip_norm=5.0)
    s.update(raw)
    for name in ("epochs", "batch_size", "cpu_threads"):
        if type(s[name]) is not int or s[name] < 1:
            raise ValueError(name + " must be a positive integer")
    for name in ("num_workers", "seed"):
        if type(s[name]) is not int or s[name] < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if s["seed"] >= 2**63:
        raise ValueError("seed must be smaller than 2**63")
    for name in ("deterministic", "amp"):
        if type(s[name]) is not bool:
            raise ValueError(name + " must be a boolean")
    _positive_number(s["learning_rate"], "learning_rate")
    _positive_number(s["weight_decay"], "weight_decay", allow_zero=True)
    if s["gradient_clip_norm"] is not None:
        _positive_number(s["gradient_clip_norm"], "gradient_clip_norm")
    if s["early_stopping_patience"] is not None:
        if type(s["early_stopping_patience"]) is not int or s["early_stopping_patience"] < 1:
            raise ValueError("early_stopping_patience must be null or a positive integer")
    schedule = s.get("scheduler", {"type": "reduce_on_plateau"})
    if schedule is None:
        schedule = {"type": "none"}
    if not isinstance(schedule, dict):
        raise ValueError("scheduler must be an object or null")
    schedule = dict(type="reduce_on_plateau", factor=0.5, patience=3, min_lr=1e-6) | schedule
    if schedule["type"] not in ("none", "reduce_on_plateau"):
        raise ValueError("scheduler.type must be reduce_on_plateau or none")
    if schedule["type"] != "none":
        _positive_number(schedule["factor"], "scheduler.factor")
        if schedule["factor"] >= 1:
            raise ValueError("scheduler.factor must be smaller than one")
        if type(schedule["patience"]) is not int or schedule["patience"] < 0:
            raise ValueError("scheduler.patience must be a nonnegative integer")
        _positive_number(schedule["min_lr"], "scheduler.min_lr", allow_zero=True)
        if schedule["min_lr"] > s["learning_rate"]:
            raise ValueError("scheduler.min_lr must not exceed learning_rate")
    s["scheduler"] = schedule
    return s


def _make_dataset(samples, dataset_root, config, training=False):
    # Lazy imports keep prepare-only and trainer unit tests independent of codecs.
    from vision.dataset import VideoGestureDataset
    return VideoGestureDataset(samples, dataset_root, config, training=training)


def _make_model(config, classes):
    from vision.models import build_model
    return build_model(config, classes)


def _model_info(model, config):
    from vision.models import model_info
    return dict(name=config.get("model"), class_name=type(model).__name__,
                model_parameters=config.get("model_parameters", {}),
                provenance=model_info(config),
                modality=config["modality"], views=config.get("views"), input=config.get("input"),
                trainable_parameters=sum(p.numel() for p in model.parameters()))


def _loader(dataset, settings, device, shuffle=False):
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=_seed_worker, drop_last=False,
                      generator=torch.Generator().manual_seed(settings["seed"]))


def _check_logits(logits, targets, classes):
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(targets), classes):
        raise ValueError("The model must return logits shaped [batch, number_of_classes]")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("The model produced non-finite logits")


def evaluate(model, loader, label_mapping, device, amp=False):
    """Evaluate one split in loader order; do not choose or change checkpoints."""
    model.eval()
    total_loss, actual, predicted = 0.0, [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inputs)
                _check_logits(logits, targets, len(label_mapping))
                loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            total_loss += loss.item()
            actual.extend(targets.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    metrics = classification_metrics(actual, predicted, label_mapping)
    metrics["loss"] = total_loss / len(actual)
    return metrics, predicted


def _checkpoint(output, filename, model, optimizer, scheduler, scaler, epoch,
                label_mapping, config, selection, environment, model_info):
    payload = dict(format_version=1, model_state_dict=model.state_dict(),
                   optimizer_state_dict=optimizer.state_dict(), epoch=epoch,
                   scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                   scaler_state_dict=scaler.state_dict(), label_mapping=label_mapping,
                   config=config, selection=selection, model_info=model_info,
                   environment=environment)
    path = _artifact(output, filename)
    temporary = _artifact(output, filename + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_test_outputs(output, samples, metrics, predictions, label_mapping):
    _write_json(output, "test_metrics.json", metrics)
    names = metrics["class_names"]
    with _artifact(output, "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_gesture / predicted_gesture"] + names)
        writer.writerows([name] + row for name, row in zip(names, metrics["confusion_matrix"]))
    inverse = {value: key for key, value in label_mapping.items()}
    with _artifact(output, "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["sample_id", "parent_record_id", "subject_id", "session_id", "object_id", "repetition",
                  "target", "prediction", "target_gesture", "predicted_gesture"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for sample, prediction in zip(samples, predictions):
            target = int(sample["label"])
            row = {key: sample.get(key, "") for key in fields[:6]}
            row.update(target=target, prediction=prediction, target_gesture=inverse[target],
                       predicted_gesture=inverse[prediction])
            writer.writerow(row)


def train_fold(fold, label_mapping, config, output_dir, dataset_root):
    """Optimize one independent fold and evaluate its held-out test set once."""
    validate_label_mapping(label_mapping)
    if config.get("modality") not in ("rgb", "depth"):
        raise ValueError("modality must be rgb or depth")
    settings = _settings(config)
    device = choose_device(settings["device"])
    samples = fold["splits"]
    for part in ("train", "validation", "test"):
        if not isinstance(samples.get(part), list):
            raise ValueError("Each fold partition must be a list of prepared samples")
        for sample in samples[part]:
            label = sample.get("label")
            if type(label) is not int or not 0 <= label < len(label_mapping):
                raise ValueError("Prepared samples need an in-range integer label")
    if not samples["train"] or not samples["test"]:
        raise ValueError("Training and test partitions must be nonempty")
    output = _output_directory(output_dir, config["modality"])
    torch.set_num_threads(settings["cpu_threads"])
    set_seed(settings["seed"], settings["deterministic"])
    amp = settings["amp"] and device.type == "cuda"
    has_validation = bool(samples["validation"])
    selection = "validation_macro_f1" if has_validation else "last_epoch_no_validation"
    loaders = {part: _loader(_make_dataset(part_samples, dataset_root, config, training=part == "train"),
                             settings, device, shuffle=part == "train")
               for part, part_samples in samples.items() if part in ("train", "validation") and part_samples}
    model = _make_model(config, len(label_mapping)).to(device)
    info = _model_info(model, config)
    info.update(loss="cross_entropy", optimizer="AdamW", checkpoint_selection=selection)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    schedule = settings["scheduler"]
    scheduler = None
    if has_validation and schedule["type"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=schedule["factor"],
                        patience=schedule["patience"], min_lr=schedule["min_lr"])
    info["scheduler"] = dict(schedule, active=scheduler is not None, monitor="validation_loss" if scheduler else None)
    if not has_validation:
        info["scheduler"]["disabled_reason"] = "No validation split; no test information is used for scheduling"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    environment = environment_info(device, amp)
    _write_json(output, "environment.json", environment)
    _write_json(output, "model.json", info)
    save_prepared_fold(fold, output)
    started = time.monotonic()
    history, best_score, best_loss, best_epoch, stale = [], -1.0, float("inf"), 0, 0
    print("{}: {} on {} | {} | views {} | train/validation/test {}/{}/{}".format(
        fold["name"], config.get("model"), device, config["modality"], config.get("views"),
        len(samples["train"]), len(samples["validation"]), len(samples["test"])), flush=True)
    with _artifact(output, "history.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["epoch", "learning_rate", "next_learning_rate", "train_loss", "train_accuracy",
                  "validation_loss", "validation_accuracy", "validation_macro_f1", "epoch_seconds"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for epoch in range(1, settings["epochs"] + 1):
            epoch_started = time.monotonic()
            model.train()
            total_loss, correct, count = 0.0, 0, 0
            epoch_lr = optimizer.param_groups[0]["lr"]
            for inputs, targets in loaders["train"]:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(inputs)
                    _check_logits(logits, targets, len(label_mapping))
                    loss = nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; review model input/preprocessing")
                scaler.scale(loss).backward()
                if settings["gradient_clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                    # AMP may encounter an overflow while finding its initial
                    # scale. GradScaler detects it and skips that optimizer
                    # update; non-AMP non-finite gradients remain a hard error.
                    nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(targets)
                correct += int((logits.detach().argmax(1) == targets).sum().item())
                count += len(targets)
            row = dict(epoch=epoch, learning_rate=epoch_lr, train_loss=total_loss / count,
                       train_accuracy=correct / count, validation_loss="", validation_accuracy="", validation_macro_f1="")
            improved = False
            if has_validation:
                validation, _ = evaluate(model, loaders["validation"], label_mapping, device, amp)
                row.update(validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                           validation_macro_f1=validation["macro_f1"])
                improved = validation["macro_f1"] > best_score or (
                    validation["macro_f1"] == best_score and validation["loss"] < best_loss)
                if improved:
                    best_score, best_loss, best_epoch, stale = validation["macro_f1"], validation["loss"], epoch, 0
                else:
                    stale += 1
                if scheduler:
                    scheduler.step(validation["loss"])
            if improved:
                _checkpoint(output, "best.pt", model, optimizer, scheduler, scaler, epoch, label_mapping,
                            config, selection, environment, info)
            _checkpoint(output, "last.pt", model, optimizer, scheduler, scaler, epoch, label_mapping,
                        config, selection, environment, info)
            row.update(next_learning_rate=optimizer.param_groups[0]["lr"], epoch_seconds=time.monotonic()-epoch_started)
            history.append(row)
            writer.writerow(row)
            f.flush()
            val_text = ", validation loss {:.4f}, F1 {:.3f}".format(row["validation_loss"], row["validation_macro_f1"]) if has_validation else ""
            print("{} epoch {}/{}: train loss {:.4f}, accuracy {:.3f}{} | lr {:.3g} | {:.1f}s".format(
                fold["name"], epoch, settings["epochs"], row["train_loss"], row["train_accuracy"], val_text,
                row["learning_rate"], row["epoch_seconds"]), flush=True)
            if row["next_learning_rate"] < row["learning_rate"]:
                print("Learning rate reduced to {:.3g} for the next epoch".format(row["next_learning_rate"]), flush=True)
            if has_validation and settings["early_stopping_patience"] is not None and stale >= settings["early_stopping_patience"]:
                print("Early stopping: validation checkpoint score did not improve for {} epochs".format(stale), flush=True)
                break
    selected_name = "best.pt" if has_validation else "last.pt"
    payload = torch.load(_artifact(output, selected_name), map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    # The test dataset is not instantiated or read until checkpoint selection ends.
    test_loader = _loader(_make_dataset(samples["test"], dataset_root, config), settings, device)
    test_metrics, predictions = evaluate(model, test_loader, label_mapping, device, amp)
    if len(predictions) != len(samples["test"]):
        raise ValueError("Test prediction count does not match prepared sample metadata")
    _save_test_outputs(output, samples["test"], test_metrics, predictions, label_mapping)
    result = dict(name=fold["name"], fold=fold["name"], modality=config["modality"], model=config.get("model"),
                  views=config.get("views"), epochs_completed=len(history),
                  best_epoch=best_epoch if has_validation else len(history), selected_epoch=payload["epoch"],
                  selected_checkpoint=selected_name, selection=selection, elapsed_s=time.monotonic()-started,
                  test_metrics=test_metrics)
    _write_json(output, "result.json", result)
    print(format_test_report(result), flush=True)
    return result
