"""Task/protocol checks on invented annotations; never opens the source dataset."""
from copy import deepcopy
import unittest

from scripts.artifacts import experiment_run_name
from scripts.labels import build_label_mapping
from scripts.metrics import aggregate_fold_metrics, classification_metrics
from scripts.protocols import make_object_split, validate_split
from scripts.tasks import joint_label, label_of, split_joint_label, validate_task, with_task_labels
from scripts.training_protocols import build_training_splits


P1 = dict(name="protocol1", type="leave_one_subject_out", split_before_windowing=True)
P2 = dict(name="protocol2", type="held_out_repetitions", split_before_windowing=True,
          train_repetitions=[1, 2, 3, 4], test_repetitions=[5],
          gesture_rules={
              "screw": dict(train_repetitions=[1, 2], test_repetitions=[3]),
              "insert": dict(train_repetitions=[1, 2, 3, 4, 6, 7, 8, 9], test_repetitions=[5, 10])})
P3 = dict(name="protocol3", type="held_out_objects", split_before_windowing=True)


def records():
    rows = []
    for subject in ("A", "B", "C"):
        for gesture, session, objects, repetitions in (
                ("grasp", "step1", range(1, 7), range(1, 6)),
                ("release", "step1", range(1, 7), range(1, 6)),
                ("screw", "step2", range(1, 3), range(1, 4)),
                ("insert", "step3", range(3, 6), range(1, 11))):
            for obj in objects:
                for rep in repetitions:
                    rows.append(dict(record_id=f"{subject}-{gesture}-{obj}-{rep}",
                                     subject_id=subject, session_id=session, object_id=str(obj),
                                     repetition=rep, gesture_label=gesture, intervals={}))
    return rows


def trials(rows):
    return {(r["subject_id"], r["session_id"], r["object_id"], r["repetition"]) for r in rows}


class TaskProtocolTests(unittest.TestCase):
    def test_target_encoding_preserves_original_labels_and_source_records(self):
        source = records()
        before = deepcopy(source)
        joint = with_task_labels(source, "gesture_object")
        self.assertEqual(source, before)
        self.assertEqual(label_of(joint[0]), "grasp::object=1")
        self.assertEqual(joint[0]["gesture_label"], "grasp")
        self.assertEqual(with_task_labels(joint)[0]["target_label"], "grasp")
        for gesture, obj in (("grasp::object=two", "x/y"), ("screw", "1")):
            self.assertEqual(split_joint_label(joint_label(gesture, obj)), (gesture, obj))
        with self.assertRaises(ValueError):
            validate_task("unknown")
        with self.assertRaisesRegex(ValueError, "Protocol 3"):
            with_task_labels(source, "gesture_object", P3)
        with self.assertRaises(ValueError):
            build_training_splits(joint, P3)

    def test_protocol3_last_objects_and_whole_trials(self):
        source = records()
        before = deepcopy(source)
        fold, = build_training_splits(source, P3)
        again, = build_training_splits(list(reversed(source)), P3)
        self.assertEqual(source, before)
        self.assertEqual(fold["metadata"]["test_object_ids"],
                         {"grasp": ["6"], "release": ["6"], "screw": ["2"], "insert": ["5"]})
        self.assertEqual(trials(fold["validation"]), trials(again["validation"]))
        for gesture in ("grasp", "release", "screw", "insert"):
            objects = [{r["object_id"] for r in fold[p] if r["gesture_label"] == gesture}
                       for p in ("train", "validation", "test")]
            self.assertTrue(all(objects))
            self.assertFalse((objects[0] | objects[1]) & objects[2])
        self.assertFalse(trials(fold["train"]) & trials(fold["validation"]))
        self.assertFalse(trials(fold["train"] + fold["validation"]) & trials(fold["test"]))
        self.assertEqual(sum(len(fold[p]) for p in ("train", "validation", "test")), len(source))

    def test_object_order_and_override_validation(self):
        from scripts.protocols import SplitValidationError
        source = records()
        for row in source:
            row["intervals"] = {"signal": {"start_s": 0.0, "end_s": 1.0}}
        with self.assertRaises(SplitValidationError):
            make_object_split(source)
        warned = make_object_split(source, allow_within_subject_test_overlap=True)
        self.assertTrue(warned["valid"])
        self.assertTrue(any(i["code"] == "overlapping_intervals" and i["severity"] == "warning"
                            for i in warned["issues"]))
        source = [dict(r, object_id="10" if r["object_id"] == "6" else r["object_id"])
                  for r in records()]
        fold = make_object_split(source)
        self.assertEqual(fold["metadata"]["test_object_ids"]["grasp"], ["10"])
        fold = make_object_split(records(), test_object_ids={"grasp": ["6"], "release": ["6"], "screw": ["1"]})
        self.assertEqual(fold["metadata"]["test_object_ids"]["screw"], ["1"])
        for override in ({"missing": ["1"]}, {"screw": ["9"]}, {"screw": ["1", "2"]}, {"screw": "2"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                make_object_split(records(), test_object_ids=override)
        with self.assertRaises(ValueError):
            make_object_split([r for r in records() if r["object_id"] == "1"])

    def test_joint_class_coverage_and_repetition_rules_for_protocols1_and2(self):
        source = with_task_labels(records(), "gesture_object")
        all_labels = {label_of(r) for r in source}
        for protocol in (P1, P2):
            for fold in build_training_splits(source, protocol):
                for part in ("train", "validation", "test"):
                    self.assertEqual({label_of(r) for r in fold[part]}, all_labels)
                self.assertFalse(trials(fold["train"]) & trials(fold["validation"]))
                if protocol == P2:
                    self.assertEqual({r["repetition"] for r in fold["test"] if r["gesture_label"] == "screw"}, {3})
                    self.assertEqual({r["repetition"] for r in fold["test"] if r["gesture_label"] == "insert"}, {5, 10})
        fold = build_training_splits(source, P2)[0]
        missing = "grasp::object=1"
        fold["excluded"] += [r for r in fold["test"] if label_of(r) == missing]
        fold["test"] = [r for r in fold["test"] if label_of(r) != missing]
        self.assertTrue(any(i["code"] == "missing_class_coverage" and i.get("partition") == "test"
                            for i in validate_split(fold)))

    def test_joint_metrics_separate_object_and_gesture_errors(self):
        mapping = build_label_mapping(["grasp::object=1", "grasp::object=2", "release::object=1"])
        metrics = classification_metrics([0, 1, 2], [1, 1, 0], mapping)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        for component in ("gesture", "object"):
            self.assertAlmostEqual(metrics["component_metrics"][component]["accuracy"], 2 / 3)
        summary = aggregate_fold_metrics([metrics, metrics])
        self.assertAlmostEqual(summary["component_metrics"]["gesture"]["accuracy"]["mean"], 2 / 3)
        self.assertNotIn("component_metrics", classification_metrics([0], [0], {"grasp": 0}))

    def test_joint_run_name_is_distinct(self):
        config = dict(model="TCN", training=dict(epochs=1))
        before = deepcopy(config)
        gesture = experiment_run_name("imu", config, P2)
        joint = experiment_run_name("imu", dict(config, task="gesture_object"), P2)
        self.assertNotEqual(gesture, joint)
        self.assertTrue(joint.startswith("imu_gesture_object_TCN_batch"))
        self.assertNotIn("protocol2_gesture_object", joint)
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
