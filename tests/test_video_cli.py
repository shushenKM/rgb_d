"""Check portable entrypoints, output boundaries, and modality/config overrides."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import video_cli
from scripts.artifacts import PROJECT_ROOT, create_run_directory


class VideoCliTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT_ROOT / "runs" / "rgb"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="unittest_cli_", dir=base)
        self.run = Path(self.temporary.name)
        self.config = json.loads((PROJECT_ROOT / "configs/rgb.json").read_text())
        self.dataset = json.loads((PROJECT_ROOT / "configs/dataset.json").read_text())
        self.protocol = json.loads((PROJECT_ROOT / "configs/protocol2.json").read_text())

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepare_overrides_are_saved_without_training(self):
        sample = {"sample_id": "example", "label": 0}
        fold = {"name": "protocol2", "splits": {p: [sample] for p in ("train", "validation", "test")}}
        prepared = {"valid": True, "folds": [fold], "label_mapping": {"example": 0},
                    "issues": [], "manifest": [], "dataset_root": PROJECT_ROOT.parent / "dataset"}
        with patch.object(video_cli, "read_config", side_effect=[copy.deepcopy(self.config), self.protocol, self.dataset]), \
             patch.object(video_cli, "create_run_directory", return_value=self.run), \
             patch("vision.dataset.prepare_video_dataset", return_value=prepared) as prepare, \
             patch("vision.training.train_fold") as train, contextlib.redirect_stdout(io.StringIO()):
            result = video_cli.train_video("rgb", "unused", "unused", prepare_only=True,
                                          model="mvtn", views=["above"], frames=4, image_size=64,
                                          batch_size=2, device="cpu", epochs=1)
        self.assertEqual(result, self.run)
        train.assert_not_called()
        effective = prepare.call_args.args[1]
        self.assertEqual(effective["model"], "mvtn")
        self.assertEqual(effective["views"], ["above"])
        self.assertEqual(effective["input"], {"num_frames": 4, "image_size": 64})
        self.assertEqual(effective["training"]["batch_size"], 2)
        self.assertEqual(json.loads((self.run / "state.json").read_text())["state"], "prepared")
        self.assertTrue((self.run / "protocol2/test_samples.json").exists())

    def test_read_only_and_entrypoint_mismatch_rejected_before_run(self):
        for config, dataset in [(dict(self.config, modality="depth"), self.dataset),
                                (self.config, dict(self.dataset, read_only=False))]:
            with patch.object(video_cli, "read_config", side_effect=[config, self.protocol, dataset]), \
                 patch.object(video_cli, "create_run_directory") as create:
                with self.assertRaises(ValueError):
                    video_cli.train_video("rgb", "unused", "unused")
                create.assert_not_called()

    def test_subject_override_rejected_for_repetition_protocol(self):
        with patch.object(video_cli, "read_config", side_effect=[self.config, self.protocol, self.dataset]), \
             patch.object(video_cli, "create_run_directory") as create:
            with self.assertRaisesRegex(ValueError, "only to protocol1"):
                video_cli.train_video("rgb", "unused", "unused", test_subject="1Alice")
            create.assert_not_called()

    def test_run_paths_reject_dataset_and_other_modality(self):
        for value in ("../dataset/generated", "runs/depth/other", "runs/rgb/../../dataset"):
            with self.assertRaises(ValueError):
                create_run_directory(value, "forbidden", modality="rgb")

    def test_default_protocols_and_user_overlap_choice(self):
        self.assertEqual(self.protocol["train_repetitions"], [1, 2, 3, 4])
        self.assertEqual(self.protocol["test_repetitions"], [5])
        self.assertEqual(self.protocol["gesture_rules"]["screw"]["test_repetitions"], [3])
        for gesture in ("insert", "deinsert"):
            self.assertEqual(self.protocol["gesture_rules"][gesture]["test_repetitions"], [5, 10])
        for modality in ("rgb", "depth"):
            config = json.loads((PROJECT_ROOT / "configs" / (modality + ".json")).read_text())
            self.assertIs(config["data_quality"]["allow_within_subject_test_overlap"], True)
        self.assertEqual(self.dataset["dataset_root"], "../dataset")


if __name__ == "__main__":
    unittest.main()
