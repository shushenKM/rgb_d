"""Put final scores first in saved JSON and show equal-fold component averages.

Scores remain fractions (0--1); terminal output uses percentages. For joint
classification, the main score requires the exact pair; gesture/object scores
come from those same predictions. Confusion matrices and class detail stay in
test_metrics below final_summary. Existing readers and metric values are retained.

Run python -m scripts.result_summary --all to update existing result files, or
--run runs/<modality>/<run_name> for one run. Only generated files under runs/
are changed, atomically; no checkpoints, data, annotations, or Git operations.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

SCORES = ("accuracy", "macro_f1", "weighted_f1")


def _scores(metric):
    return {key: deepcopy(metric[key]) for key in SCORES if key in metric}


def _views(metric):
    components = metric.get("component_metrics") or {}
    joint = bool(components) or metric.get("task") == "gesture_object"
    result = {"gesture_object" if joint else "gesture": _scores(metric)}
    if joint:
        for key in ("gesture", "object"):
            if key in components:
                result[key] = _scores(components[key])
    return ("gesture_object" if joint else "gesture"), result


def summarize_result(result):
    """Return a new result with scalar conclusions first; preserve all detail."""
    metric = result["test_metrics"]
    task, scores = _views(metric)
    summary = dict(task=task, score_scale="fraction_0_to_1", **scores)
    for source, target in (("sample_count", "test_examples"), ("loss", "test_loss")):
        if source in metric:
            summary[target] = metric[source]
    for key in ("name", "selected_checkpoint", "selected_epoch", "epochs_completed", "elapsed_s"):
        if key in result:
            summary[key] = result[key]
    return dict(final_summary=summary, **{k: v for k, v in result.items() if k != "final_summary"})


def summarize_aggregate(aggregate):
    """Keep legacy fields; add equal-fold means/SDs for each available target."""
    task, scores = _views(aggregate)
    summary = dict(task=task, completed_fold_count=aggregate["fold_count"],
                   averaging="equal_weight_per_completed_test_fold",
                   standard_deviation="population_across_folds",
                   score_scale="fraction_0_to_1", **scores)
    return dict(final_summary=summary, **{k: v for k, v in aggregate.items() if k != "final_summary"})


def prepare_json(filename, value):
    """Apply only to recognized result filenames; leave all other JSON alone."""
    if filename == "result.json" and isinstance(value, dict) and "test_metrics" in value:
        return summarize_result(value)
    if filename == "results.json" and isinstance(value, list):
        return [summarize_result(row) if isinstance(row, dict) and "test_metrics" in row else row for row in value]
    if filename == "aggregate_metrics.json" and isinstance(value, dict) and "fold_count" in value:
        return summarize_aggregate(value)
    return value


def format_overall_summary(aggregate):
    """Display all targets; protocol1 folds correspond to tested participants."""
    summary = summarize_aggregate(aggregate)["final_summary"]
    count = summary["completed_fold_count"]
    lines = ["", "Final summary: {} completed test fold(s)".format(count),
             "Equal-fold mean +/- population SD (percentage points):"]
    template = "{:<15}  {:>21}  {:>21}  {:>21}"
    lines.append(template.format("Target", "Accuracy %", "Macro F1 %", "Weighted F1 %"))
    for key in ("gesture_object", "gesture", "object"):
        if key not in summary:
            continue
        cells = ["{:.2f} +/- {:.2f}".format(100 * summary[key][m]["mean"], 100 * summary[key][m]["std"])
                 for m in SCORES]
        lines.append(template.format(key, *cells))
    if count == 1:
        lines.append("One completed fold: SD is zero; this is not an all-participant average.")
    else:
        lines.append("Each completed fold has equal weight; scores are not pooled across samples.")
    return "\n".join(lines)


def refresh_saved_results(path):
    """Update saved conclusions without changing any existing metric/detail."""
    root = Path(__file__).resolve().parents[1]
    boundary = (root / "runs").resolve()
    selected = Path(path)
    selected = (selected if selected.is_absolute() else root / selected).resolve()
    if not boundary.is_relative_to(root) or not selected.is_relative_to(boundary):
        raise ValueError("Result updates must stay under this repository's runs/")
    if not selected.exists():
        raise FileNotFoundError(selected)
    names = ("result.json", "results.json", "aggregate_metrics.json")
    paths = [selected] if selected.is_file() else sorted(p for name in names for p in selected.rglob(name))
    plans = []
    for target in paths:
        if target.name not in names or not target.resolve().is_relative_to(boundary):
            raise ValueError("Only result JSON files under runs/ may be updated")
        raw = target.read_text(encoding="utf-8-sig")
        value = json.loads(raw)
        updated = prepare_json(target.name, value)
        rendered = json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if raw != rendered:
            plans.append((target, raw, rendered))
    for target, raw, rendered in plans:
        if target.read_text(encoding="utf-8-sig") != raw:
            raise RuntimeError("Result changed during refresh; retry when training is finished: " + str(target))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return dict(inspected=len(paths), updated=len(plans))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Refresh saved JSON results under this project's runs/")
    group.add_argument("--run", help="One existing run/fold directory or result JSON under runs/")
    args = parser.parse_args()
    result = refresh_saved_results("runs" if args.all else args.run)
    print("Result summaries: inspected {inspected}, updated {updated} JSON files.".format(**result))


if __name__ == "__main__":
    main()
