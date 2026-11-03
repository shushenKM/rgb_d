"""Verify descriptive names and preservation of existing experiment outputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from scripts.artifacts import PROJECT_ROOT, create_run_directory, experiment_run_name

class RunNameTests(unittest.TestCase):
    def test_effective_settings_and_modes(self):
        config = {"model": "mvtn", "views": ["above", "front"], "training": {"batch_size": 8, "epochs": 50, "learning_rate": .0001, "seed": 7}, "input": {"num_frames": 16, "image_size": 224}}
        name = experiment_run_name("rgb", config, {"name": "protocol1", "test_subject_id": "1Alice"}, prepare_only=True)
        self.assertEqual(name, "rgb_mvtn_batch8_epoch50_protocol1_above-front_frames16_size224_lr0.0001_seed7_test-1Alice_prepare")
        for modality, key, selected in (("imu", "sensor_ids", ["E45A", "E477"]), ("semg", "grid_ids", ["grid2"])):
            name = experiment_run_name(modality, {key: selected, "input": {"num_points": 64}}, {"name": "protocol2"})
            self.assertTrue(name.startswith(modality + "_"))
            self.assertIn("_batch64_epoch30_protocol2_" + "-".join(selected) + "_points64_", name)
        self.assertTrue(experiment_run_name("depth", config, {"name": "smoke"}, synthetic=True).endswith("_synthetic"))

    def test_portable_bounded_names(self):
        name = experiment_run_name("imu", {"model": "a/b:c " * 100}, {"name": "protocol2"})
        self.assertLessEqual(len(name), 160)
        self.assertRegex(name, r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def test_collision_concurrency_and_output_protection(self):
        base = PROJECT_ROOT / "runs"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="unittest_run_names_", dir=base) as temporary:
            root = Path(temporary)
            first = create_run_directory(project_root=root, default_name="imu_model_batch64_epoch50")
            marker = first / "result.txt"
            marker.write_text("keep", encoding="utf-8")
            second = create_run_directory(project_root=root, default_name=first.name)
            self.assertEqual(second.name, first.name + "_run02")
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda _: create_run_directory(project_root=root, default_name=first.name), range(4)))
            self.assertEqual({p.name for p in paths}, {first.name + "_run%02d" % i for i in range(3, 7)})
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            with self.assertRaises(FileExistsError):
                create_run_directory(name=first.name, project_root=root)
            for unsafe in ("../escape", "a/b", ""):
                with self.assertRaises(ValueError):
                    create_run_directory(name=unsafe, project_root=root)
            with self.assertRaises(ValueError):
                create_run_directory("../dataset", project_root=root)
            blocked = root / "runs/imu/blocked"
            blocked.write_text("file", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_run_directory("runs/imu/blocked", project_root=root)

if __name__ == "__main__":
    unittest.main()
