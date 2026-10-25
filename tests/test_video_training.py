"""Exercise optimization, checkpoint selection, scheduler and output isolation."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from vision import training


class InventedClips(Dataset):
    def __init__(self, samples, _root, _config, training=False):
        self.samples = samples
        self.partition = samples[0]["partition"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        target = self.samples[index]["label"]
        return torch.full((1, 2, 3, 4, 4), float(target) + 0.2), target


class TinyVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, x):
        return self.classifier(x.mean(dim=(1, 2, 3, 4, 5)).unsqueeze(1))


def make_fold(validation=True):
    splits = {}
    for part, count in [("train", 4), ("validation", 2 if validation else 0), ("test", 2)]:
        splits[part] = [dict(sample_id=f"{part}_{i}", parent_record_id=f"parent_{part}_{i}",
                            subject_id=part, session_id="step1", object_id="1", repetition=i + 1,
                            label=i % 2, gesture_label="alpha" if i % 2 == 0 else "beta", partition=part)
                        for i in range(count)]
    return dict(name="invented", splits=splits, split_summary={p: len(s) for p, s in splits.items()},
                protocol_metadata={"synthetic": True})


def make_config():
    return dict(modality="rgb", model="tsm", views=["front"], input={"num_frames": 2, "image_size": 4},
                training=dict(epochs=2, batch_size=2, learning_rate=0.01, weight_decay=0,
                              device="cpu", cpu_threads=1, num_workers=0, amp=False,
                              seed=7, deterministic=True,
                              scheduler={"type": "reduce_on_plateau", "factor": 0.5, "patience": 0, "min_lr": 1e-6}))


class VideoTrainingTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(training.PROJECT_ROOT) / "runs" / "rgb"
        self.base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="unittest_training_", dir=self.base)
        self.output = Path(self.temporary.name) / "fold"
        self.mapping = {"alpha": 0, "beta": 1}
        self.config = make_config()
        self.fold = make_fold()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(training, "_make_dataset", side_effect=InventedClips))
        self.stack.enter_context(patch.object(training, "_make_model", side_effect=lambda *_: TinyVideoModel()))
        self.stack.enter_context(patch.object(training, "_model_info", return_value={"synthetic": True}))

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def run_training(self, fold=None, config=None):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = training.train_fold(fold or self.fold, self.mapping, config or self.config,
                                         self.output, Path(self.temporary.name) / "unused_source")
        return result, stdout.getvalue()

    def test_artifacts_and_terminal_report(self):
        result, terminal = self.run_training()
        for name in ("best.pt", "last.pt", "history.csv", "test_metrics.json", "confusion_matrix.csv",
                     "predictions.csv", "environment.json", "model.json", "result.json", "test_samples.json",
                     "train_samples.json", "validation_samples.json", "protocol.json", "split_summary.json"):
            self.assertTrue((self.output / name).is_file(), name)
        self.assertIn("Final test results: invented", terminal)
        self.assertIn("Confusion matrix", terminal)
        self.assertEqual(result["test_metrics"]["sample_count"], 2)
        with (self.output / "predictions.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["sample_id"] for r in rows], ["test_0", "test_1"])
        self.assertEqual(rows[0]["parent_record_id"], "parent_test_0")
        payload = torch.load(self.output / "best.pt", weights_only=True)
        self.assertEqual(payload["label_mapping"], self.mapping)
        self.assertEqual(payload["config"]["modality"], "rgb")

    def test_selection_uses_validation_and_test_is_only_evaluated_once(self):
        original = training.evaluate
        calls, states = [], []

        def controlled(model, loader, *args, **kwargs):
            part = loader.dataset.partition
            calls.append(part)
            states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            metrics, predictions = original(model, loader, *args, **kwargs)
            if part == "validation":
                number = calls.count("validation")
                metrics.update(macro_f1=0.9 if number == 1 else 0.8, loss=float(number))
            return metrics, predictions

        self.config["training"].update(epochs=5, early_stopping_patience=1)
        with patch.object(training, "evaluate", side_effect=controlled):
            result, terminal = self.run_training()
        self.assertEqual(calls, ["validation", "validation", "test"])
        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(result["selected_epoch"], 1)
        self.assertTrue(all(torch.equal(states[0][k], states[-1][k]) for k in states[0]))
        self.assertTrue(any(not torch.equal(states[0][k], states[1][k]) for k in states[0]))
        self.assertIn("Learning rate reduced to 0.005", terminal)
        with (self.output / "history.csv").open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        self.assertEqual(float(history[-1]["next_learning_rate"]), 0.005)
        self.assertIn("Early stopping", terminal)

    def test_no_validation_selects_last_and_disables_scheduler(self):
        original = training.evaluate
        with patch.object(training, "evaluate", wraps=original) as evaluator:
            result, _ = self.run_training(make_fold(False))
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(result["selected_checkpoint"], "last.pt")
        self.assertEqual(result["selected_epoch"], 2)
        self.assertEqual(result["selection"], "last_epoch_no_validation")
        info = json.loads((self.output / "model.json").read_text())
        self.assertFalse(info["scheduler"]["active"])
        self.assertFalse((self.output / "best.pt").exists())

    def test_rejects_outside_run_outputs_before_creating_them(self):
        outside = Path(training.PROJECT_ROOT).parent / "dataset" / "must_never_be_created"
        with self.assertRaises(ValueError):
            training.save_prepared_fold(self.fold, outside)
        self.assertFalse(outside.exists())
        with self.assertRaises(ValueError):
            training._output_directory(self.output, "depth")

    def test_invalid_settings_rejected_before_dataset_is_read(self):
        for name, value in [("epochs", 0), ("batch_size", True), ("num_workers", -1),
                            ("learning_rate", float("nan")), ("amp", "yes")]:
            with self.subTest(name=name), patch.object(training, "_make_dataset") as dataset:
                config = make_config()
                config["training"][name] = value
                with self.assertRaises(ValueError):
                    self.run_training(config=config)
                dataset.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_scheduler_rejected(self):
        for key, value in [("type", "unknown"), ("factor", 1.0), ("patience", -1), ("min_lr", 100)]:
            with self.subTest(key=key):
                config = make_config()
                config["training"]["scheduler"][key] = value
                with self.assertRaises(ValueError):
                    training._settings(config)

    def test_cuda_unavailable_message(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(training.choose_device("auto").type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA was requested but is unavailable"):
                training.choose_device("cuda")

    def test_nonfinite_logits_are_rejected(self):
        class BadModel(TinyVideoModel):
            def forward(self, x):
                return super().forward(x) * float("nan")
        with patch.object(training, "_make_model", return_value=BadModel()):
            with self.assertRaises(FloatingPointError):
                self.run_training()


if __name__ == "__main__":
    unittest.main()
