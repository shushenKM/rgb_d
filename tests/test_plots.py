"""Check reports against saved metrics, missing history, and output boundaries."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plots
from scripts.metrics import classification_metrics


class PlotTests(unittest.TestCase):
    def setUp(self):
        (plots.PROJECT_ROOT / "runs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unittest_plots_", dir=plots.PROJECT_ROOT / "runs")
        self.run = Path(self.temp.name)
        self.fold = self.run / "protocol2"
        self.fold.mkdir()
        metrics = classification_metrics([0, 0, 1], [0, 1, 1], {"approach": 0, "grasp": 1, "absent": 2})
        metrics["loss"] = .4
        self.result = dict(name="protocol2", selected_epoch=1, selected_checkpoint="best.pt",
                           epochs_completed=2, test_metrics=metrics)
        (self.fold / "result.json").write_text(json.dumps(self.result), encoding="utf-8")
        with (self.fold / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, ["epoch", "train_loss", "train_accuracy", "validation_loss",
                                       "validation_accuracy", "validation_macro_f1", "learning_rate"])
            writer.writeheader()
            for epoch in (1, 2):
                writer.writerow(dict(epoch=epoch, train_loss=1 / epoch, train_accuracy=.4 * epoch,
                                     validation_loss=.7, validation_accuracy=.6, validation_macro_f1=.5,
                                     learning_rate=.001 / epoch))

    def tearDown(self):
        self.temp.cleanup()

    def test_normalization_preserves_orientation_and_handles_absent_classes(self):
        self.assertEqual(plots.normalized_confusion([[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
                         [[50, 50, 0], [0, 100, 0], [0, 0, 0]])

    def test_render_legacy_history_and_multi_fold_report_without_changing_metrics(self):
        before = {p.name: p.read_bytes() for p in self.fold.iterdir()}
        report = plots.generate_fold_report(self.fold)
        self.assertTrue(report.is_file())
        from PIL import Image
        images = list((self.fold / "plots").glob("*.png"))
        self.assertEqual(len(images), 9)
        self.assertEqual(len(list((self.fold / "plots").glob("*.svg"))), 9)
        for path in images:
            with Image.open(path) as image:
                self.assertGreater(image.width, 500)
                image.verify()
        for name, content in before.items():
            self.assertEqual((self.fold / name).read_bytes(), content)
        other = self.run / "fold2"
        other.mkdir()
        (other / "report.html").write_text("example", encoding="utf-8")
        plots.generate_run_report(self.run, [self.result, dict(self.result, name="fold2")])
        self.assertTrue((self.run / "plots/fold_comparison.svg").is_file())
        page = (self.run / "report.html").read_text(encoding="utf-8")
        self.assertIn("protocol2/report.html", page)
        self.assertIn("fold2/report.html", page)

    def test_missing_validation_and_old_duration_are_not_invented(self):
        rows = [{"epoch": "1", "train_loss": ".5", "validation_loss": ""}]
        captured = []
        with patch.object(plots, "_save", side_effect=lambda fig, *_: captured.append(fig)):
            plots._curve(rows, [("validation_loss", "Validation", "blue")], "Loss", "Loss", 1, self.fold, "loss")
        ax = captured[0].axes[0]
        self.assertEqual(len(ax.lines), 0)
        self.assertIn("Not recorded", ax.texts[0].get_text())

    def test_plot_failure_is_separate_and_paths_cannot_escape_runs(self):
        with patch.object(plots, "generate_fold_report", side_effect=ImportError("matplotlib missing")), contextlib.redirect_stderr(io.StringIO()) as terminal:
            self.assertIsNone(plots.save_training_report(self.fold))
        self.assertIn("matplotlib missing", (self.fold / "plot_error.txt").read_text())
        self.assertFalse((self.fold / "plot_status.json").exists())
        self.assertIn("Training results are saved", terminal.getvalue())
        self.assertTrue((self.fold / "result.json").is_file())
        with self.assertRaises(ValueError):
            plots.generate_fold_report(plots.PROJECT_ROOT.parent / "dataset")
        with self.assertRaises(ValueError):
            plots._target(self.run, "../escape")


if __name__ == "__main__":
    unittest.main()
