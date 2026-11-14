"""Summary-first JSON preserves detailed metrics and averages each fold equally."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.metrics import classification_metrics, aggregate_fold_metrics
from scripts.result_summary import prepare_json, format_overall_summary, refresh_saved_results
from scripts.artifacts import write_json

ROOT = Path(__file__).resolve().parents[1]


def result():
    metric = classification_metrics([0, 1, 2], [1, 1, 1],
        {"grasp::object=1": 0, "grasp::object=2": 1, "insert::object=2": 2})
    metric["loss"] = .5
    return dict(name="protocol1_alice", selected_checkpoint="best.pt", selected_epoch=2,
                epochs_completed=3, elapsed_s=9, test_metrics=metric)


class ResultSummaryTests(unittest.TestCase):
    def test_joint_summary_is_first_and_has_no_matrix_or_class_detail(self):
        original = result()
        before = deepcopy(original)
        updated = prepare_json("result.json", original)
        self.assertEqual(next(iter(updated)), "final_summary")
        summary = updated["final_summary"]
        self.assertAlmostEqual(summary["gesture_object"]["accuracy"], 1/3)
        self.assertAlmostEqual(summary["gesture"]["accuracy"], 2/3)
        self.assertAlmostEqual(summary["object"]["accuracy"], 2/3)
        self.assertEqual(summary["test_examples"], 3)
        self.assertEqual(summary["test_loss"], .5)
        self.assertNotIn("confusion_matrix", json.dumps(summary))
        self.assertNotIn("per_class", json.dumps(summary))
        self.assertEqual({k:v for k,v in updated.items() if k!="final_summary"}, original)
        self.assertEqual(original, before)
        self.assertEqual(prepare_json("result.json", updated), updated)

    def test_gesture_only_has_no_invented_object_scores(self):
        metric = classification_metrics([0, 1], [0, 0], {"grasp":0, "insert":1})
        summary = prepare_json("result.json", dict(test_metrics=metric))["final_summary"]
        self.assertEqual(summary["task"], "gesture")
        self.assertEqual(set(summary) & {"gesture","gesture_object","object"}, {"gesture"})

    def test_aggregate_and_terminal_include_all_components_equal_fold_weight(self):
        a = result()["test_metrics"]
        b = deepcopy(a)
        a["sample_count"], b["sample_count"] = 1, 999
        b["accuracy"] = 1
        b["component_metrics"]["gesture"]["accuracy"] = 1
        b["component_metrics"]["object"]["accuracy"] = 0
        aggregate = aggregate_fold_metrics([a,b])
        summary = prepare_json("aggregate_metrics.json", aggregate)["final_summary"]
        self.assertEqual(summary["completed_fold_count"], 2)
        self.assertAlmostEqual(summary["gesture_object"]["accuracy"]["mean"], 2/3)
        self.assertAlmostEqual(summary["gesture"]["accuracy"]["mean"], 5/6)
        self.assertAlmostEqual(summary["object"]["accuracy"]["mean"], 1/3)
        self.assertAlmostEqual(summary["object"]["accuracy"]["std"], 1/3)
        text = format_overall_summary(aggregate)
        for target in ("gesture_object", "gesture", "object"):
            self.assertIn(target, text)
        self.assertIn("83.33", text)
        self.assertIn("equal weight", text)
        self.assertEqual(prepare_json("aggregate_metrics.json", aggregate)["accuracy"], aggregate["accuracy"])

    def test_writer_handles_result_list_aggregate_and_other_files(self):
        (ROOT/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="summary_test_",dir=ROOT/"runs") as folder:
            for name,value in (("result.json",result()),("results.json",[result()]),
                               ("aggregate_metrics.json",aggregate_fold_metrics([result()["test_metrics"]])),
                               ("config.json",{"unchanged":True})):
                path=Path(folder)/name
                write_json(path,value)
                saved=json.loads(path.read_text(encoding="utf-8"))
                if name=="config.json":
                    self.assertEqual(saved,value)
                else:
                    self.assertEqual(next(iter(saved[0] if isinstance(saved,list) else saved)),"final_summary")
            self.assertEqual(refresh_saved_results(folder)["updated"],0)

    def test_refresh_old_results_preserves_values_and_is_idempotent(self):
        (ROOT/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="summary_test_",dir=ROOT/"runs") as folder:
            path=Path(folder)/"result.json"
            original=result()
            path.write_text(json.dumps(original),encoding="utf-8")
            self.assertEqual(refresh_saved_results(folder),dict(inspected=1,updated=1))
            updated=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(updated["test_metrics"],original["test_metrics"])
            before=path.read_bytes()
            self.assertEqual(refresh_saved_results(folder)["updated"],0)
            self.assertEqual(path.read_bytes(),before)
        with self.assertRaises(ValueError):
            refresh_saved_results(ROOT.parent/"dataset")


if __name__=="__main__":
    unittest.main()
