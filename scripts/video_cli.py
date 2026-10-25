"""Shared command-line orchestration for independent RGB and depth experiments.

The modality entry points read portable configurations, build parent-trial
splits, save preparation manifests, then train one independent model per fold.
--prepare-only validates data/splits and writes metadata without optimization.
Every run has its own config, state, class vocabulary, split manifests and final
metrics. A failed run retains diagnostic reports. This module never writes to
source recordings or annotation folders and never starts training on import.
"""
import argparse
import copy
import sys

from scripts.artifacts import PROJECT_ROOT, create_run_directory, fold_directory, read_config, write_json


def train_video(modality, config_path, protocol_path, *, prepare_only=False,
                model=None, views=None, subjects=None, test_subject=None,
                validation_subject=None, epochs=None, device=None, batch_size=None,
                frames=None, image_size=None, workers=None, backbone_weights=None,
                run_name=None):
    """Prepare and optionally train the requested modality; return its run path."""
    if modality not in ("rgb", "depth"):
        raise ValueError("modality must be rgb or depth")
    config = copy.deepcopy(read_config(config_path))
    protocol = copy.deepcopy(read_config(protocol_path))
    if config.get("modality") != modality:
        raise ValueError("This entry point requires a {} configuration".format(modality))
    dataset = read_config(config.get("dataset_config", "configs/dataset.json"))
    if dataset.get("read_only") is not True:
        raise ValueError("dataset.read_only must remain true")
    if model is not None:
        config["model"] = model
    if views is not None:
        config["views"] = list(views)
    if subjects is not None:
        config["subject_ids"] = list(subjects)
    for key, value in (("epochs", epochs), ("device", device), ("batch_size", batch_size), ("num_workers", workers)):
        if value is not None:
            config.setdefault("training", {})[key] = value
    for key, value in (("num_frames", frames), ("image_size", image_size)):
        if value is not None:
            config.setdefault("input", {})[key] = value
    if backbone_weights is not None:
        config.setdefault("model_parameters", {})["pretrained_backbone"] = backbone_weights
    if test_subject is not None:
        protocol["test_subject_id"] = test_subject
    if validation_subject is not None:
        protocol["validation_subject_id"] = validation_subject
    if protocol.get("type") != "leave_one_subject_out" and (test_subject is not None or validation_subject is not None):
        raise ValueError("--test-subject and --validation-subject apply only to protocol1")
    if config.get("model") not in ("tsm", "mvtn"):
        raise ValueError("model must be tsm or mvtn")
    run = create_run_directory(config.get("output_root", "runs/" + modality), run_name, modality=modality)
    write_json(run / "config.json", {modality: config, "dataset": dataset, "protocol": protocol})
    write_json(run / "state.json", {"state": "preparing", "synthetic": False})
    print("Preparing {} | model={} | views={} | protocol={}".format(
        modality.upper(), config["model"], ",".join(config.get("views", [])), protocol.get("name", protocol.get("type"))), flush=True)
    try:
        from vision.dataset import prepare_video_dataset
        prepared = prepare_video_dataset(dataset, config, protocol)
        write_json(run / "issues.json", prepared.get("issues", []))
        write_json(run / "recording_manifest.json", prepared.get("manifest", []))
        write_json(run / "labels.json", prepared["label_mapping"])
        warnings = [item for item in prepared.get("issues", []) if item.get("severity") == "warning"]
        if warnings:
            from collections import Counter
            counts = Counter(item["code"] for item in warnings)
            print("Preparation warnings (details in issues.json): {}".format(
                ", ".join("{}={}".format(key, value) for key, value in sorted(counts.items()))), flush=True)
        if not prepared.get("valid", True) or not prepared.get("folds"):
            raise ValueError("Preparation did not produce valid nonempty folds")
        from vision.training import save_prepared_fold
        results = []
        for fold in prepared["folds"]:
            output = fold_directory(run, fold["name"])
            save_prepared_fold(fold, output)
            counts = {part: len(fold["splits"][part]) for part in ("train", "validation", "test")}
            print("{}: train={}, validation={}, test={} gesture samples".format(
                fold["name"], counts["train"], counts["validation"], counts["test"]), flush=True)
            if not prepare_only:
                from vision.training import train_fold
                write_json(run / "state.json", {"state": "training", "synthetic": False, "fold": fold["name"]})
                results.append(train_fold(fold, prepared["label_mapping"], config, output, prepared["dataset_root"]))
                write_json(run / "results.json", results)
        if results:
            from scripts.metrics import aggregate_fold_metrics
            aggregate = aggregate_fold_metrics(r["test_metrics"] for r in results)
            write_json(run / "aggregate_metrics.json", aggregate)
            if len(results) > 1:
                print("\nFinal test results across {} folds (mean +/- standard deviation):".format(len(results)))
                for key, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
                    print("{}: {:.2%} +/- {:.2%}".format(title, aggregate[key]["mean"], aggregate[key]["std"]), flush=True)
        write_json(run / "state.json", {"state": "prepared" if prepare_only else "complete", "synthetic": False,
                                        "fold_count": len(prepared["folds"])})
        return run
    except Exception as error:
        for attribute, filename in (("issues", "issues.json"), ("manifest", "recording_manifest.json")):
            if hasattr(error, attribute):
                write_json(run / filename, getattr(error, attribute))
        if hasattr(error, "result"):
            write_json(run / "split_error.json", error.result)
        write_json(run / "state.json", {"state": "failed", "synthetic": False,
                                        "error_type": type(error).__name__, "message": str(error)})
        print("Review this run: {}".format(run.relative_to(PROJECT_ROOT)), file=sys.stderr)
        raise


def main(modality="rgb", argv=None):
    """Parse overrides for the root train_rgb.py or train_depth.py script."""
    parser = argparse.ArgumentParser(description="Prepare/train {} gesture recognition using TSM or MVTN".format(modality.upper()))
    parser.add_argument("--config", default="configs/" + modality + ".json")
    parser.add_argument("--protocol", required=True, help="configs/protocol1.json or configs/protocol2.json")
    parser.add_argument("--prepare-only", action="store_true", help="Validate recordings and splits; save manifests without model optimization")
    parser.add_argument("--model", choices=("tsm", "mvtn"))
    parser.add_argument("--views", nargs="+", choices=("front", "above", "left", "right"), help="One view or multiple views with shared model weights and averaged logits")
    parser.add_argument("--subjects", nargs="+", help="Optional subject subset; default uses configured/all subjects")
    parser.add_argument("--test-subject", help="One protocol1 fold instead of all subject folds")
    parser.add_argument("--validation-subject", help="Explicit protocol1 validation participant")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, help="Data-loader worker count; default 0 is portable on Windows")
    parser.add_argument("--frames", type=int, help="Frames sampled per gesture and camera")
    parser.add_argument("--image-size", type=int, help="Square model input resolution")
    parser.add_argument("--backbone-weights", help="Optional local standard ResNet18 state dict, relative to repository or absolute")
    parser.add_argument("--run-name", help="New directory under runs/{}; existing runs are never overwritten".format(modality))
    args = vars(parser.parse_args(argv))
    config_path, protocol_path = args.pop("config"), args.pop("protocol")
    try:
        run = train_video(modality, config_path, protocol_path, **args)
        print("Saved run: {}".format(run.relative_to(PROJECT_ROOT)), flush=True)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print("{} pipeline stopped: {}".format(modality.upper(), error), file=sys.stderr)
        if isinstance(error, ImportError):
            print("Install requirements-video.txt in your training environment.", file=sys.stderr)
        return 2
