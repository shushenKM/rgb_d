"""Protocol names and grouped ratios; invented source trials only."""
from copy import deepcopy
import unittest

from scripts.protocol_config import normalize_protocol, percentage_settings, resolve_protocol_reference
from scripts.training_protocols import build_training_splits
from scripts.tasks import label_of, with_task_labels
from scripts.artifacts import experiment_run_name, read_config


def records():
    return [dict(record_id=f"{s}-{gesture}-{obj}-{rep}", subject_id=s,
                 session_id="step2" if gesture == "screw" else "step1",
                 object_id=str(obj), repetition=rep, gesture_label=gesture, intervals={})
            for s in ("a", "b", "c", "d") for obj in (1, 2)
            for gesture in ("grasp", "release", "screw")
            for rep in range(1, 4 if gesture == "screw" else 6)]


def config():
    return dict(name="protocol2", type="held_out_repetitions", split_before_windowing=True,
                grouped_percentages=True, split_percentages=dict(train=70, validation=15, test=15),
                split_seed=42, train_repetitions=[1, 2, 3, 4], test_repetitions=[5],
                gesture_rules={"screw": dict(train_repetitions=[1, 2], test_repetitions=[3])})


class PercentageProtocolTests(unittest.TestCase):
    def test_short_names_paths_and_legacy_alias(self):
        for name in ("protocol1", "protocol2", "protocol3"):
            self.assertEqual(resolve_protocol_reference(name), ("configs/" + name + ".json", False))
            self.assertEqual(read_config(resolve_protocol_reference(name)[0])["name"], name)
        self.assertEqual(resolve_protocol_reference("protocol2.json"), ("protocol2.json", False))
        self.assertEqual(resolve_protocol_reference("custom.json"), ("custom.json", False))
        for path in ("configs/custom.json", "C:/custom/config.json", r"configs\custom.json"):
            self.assertEqual(resolve_protocol_reference(path), (path, False))
        self.assertEqual(resolve_protocol_reference("protocol2_percentages"), ("configs/protocol2.json", True))
        self.assertEqual(resolve_protocol_reference("configs/protocol2_percentages.json"), ("configs/protocol2.json", True))
        self.assertIs(normalize_protocol(dict(config(), grouped_percentages=False), force_percentages=True)["grouped_percentages"], True)

    def test_false_keeps_repetition_assignments_and_inactive_ratios_are_ignored(self):
        protocol = config()
        protocol["grouped_percentages"] = False
        legacy = {k: v for k, v in protocol.items() if k not in ("grouped_percentages", "split_percentages", "split_seed")}
        first = build_training_splits(records(), protocol)
        self.assertEqual(first, build_training_splits(records(), legacy))
        self.assertEqual(first, build_training_splits(records(), dict(protocol, split_percentages=None, split_seed="inactive")))
        self.assertNotIn("split70", experiment_run_name("imu", {}, protocol))

    def test_percentages_keep_trials_and_all_joint_classes(self):
        for task in ("gesture", "gesture_object"):
            rows = with_task_labels(records(), task)
            before = deepcopy(rows)
            protocol = config()
            fold, = build_training_splits(rows, protocol, {"strategy": "none", "fraction": .9})
            again, = build_training_splits(list(reversed(rows)), protocol)
            labels = {label_of(r) for r in rows}
            owners = {}
            for part in ("train", "validation", "test"):
                self.assertEqual({label_of(r) for r in fold[part]}, labels)
                self.assertEqual({r["record_id"] for r in fold[part]}, {r["record_id"] for r in again[part]})
                for row in fold[part]:
                    key = tuple(row[k] for k in ("subject_id", "session_id", "object_id", "repetition"))
                    self.assertEqual(owners.setdefault(key, part), part)
            self.assertTrue(any(r["repetition"] != 5 for r in fold["test"] if r["gesture_label"] != "screw"))
            self.assertTrue(any(r["repetition"] == 5 for r in fold["train"]))
            self.assertEqual(rows, before)
            self.assertEqual(protocol, config())
            self.assertEqual(sum(len(fold[p]) for p in ("train", "validation", "test")), len(rows))
            meta = fold["metadata"]["percentage_split"]
            self.assertEqual(meta["requested_percentages"], protocol["split_percentages"])
            self.assertEqual(sum(meta["trial_counts"].values()), len(owners))
            for p in ("train", "validation", "test"):
                self.assertAlmostEqual(meta["actual_annotation_percentages"][p], 100 * len(fold[p]) / len(rows))

    def test_changed_seed_ratios_and_zero_validation(self):
        first = build_training_splits(records(), config())[0]
        second = build_training_splits(records(), dict(config(), split_seed=17))[0]
        self.assertNotEqual({r["record_id"] for r in first["test"]}, {r["record_id"] for r in second["test"]})
        changed = build_training_splits(records(), dict(config(), split_percentages=dict(train=60, validation=20, test=20)))[0]
        self.assertGreater(len(changed["test"]), len(first["test"]))
        no_val = build_training_splits(records(), dict(config(), split_percentages=dict(train=80, validation=0, test=20)))[0]
        self.assertFalse(no_val["validation"])
        self.assertTrue(no_val["valid"])
        name = experiment_run_name("imu", {}, config())
        self.assertIn("split70-15-15", name)
        self.assertIn("ss42", name)

    def test_invalid_settings_and_inadequate_class_groups_fail(self):
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "grouped_percentages"):
                build_training_splits(records(), dict(config(), grouped_percentages=value))
        for values in ({}, dict(train=60, validation=20, test=10),
                       dict(train=0, validation=50, test=50), dict(train=True, validation=15, test=15),
                       dict(train=float("nan"), validation=15, test=15)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "split_percentages"):
                percentage_settings(dict(config(), split_percentages=values))
        with self.assertRaisesRegex(ValueError, "split_seed"):
            build_training_splits(records(), dict(config(), split_seed=-1))
        with self.assertRaisesRegex(ValueError, "at least 3"):
            build_training_splits(records()[:2], config())
        with self.assertRaisesRegex(ValueError, "only to protocol2"):
            normalize_protocol(dict(config(), type="held_out_objects"))
        with self.assertRaisesRegex(ValueError, "before windowing"):
            build_training_splits(records(), dict(config(), split_before_windowing=False))

    def test_accepted_overlap_is_explicit_and_does_not_change_normal_mode(self):
        rows = records()
        for r in rows:
            r["intervals"] = {"signal": dict(start_s=0., end_s=1.)}
        with self.assertRaises(ValueError):
            build_training_splits(rows, config(), {"allow_train_validation_overlap": True})
        fold, = build_training_splits(rows, dict(config(), allow_within_subject_test_overlap=True),
                                     {"allow_train_validation_overlap": True})
        self.assertTrue(fold["valid"])
        self.assertTrue(any(i["code"] == "overlapping_intervals" and i["severity"] == "warning" for i in fold["issues"]))


if __name__ == "__main__":
    unittest.main()
