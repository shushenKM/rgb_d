"""Compact run outputs: parameters and summaries, with details only on failure.

Sample/annotation lists and recording manifests stay in memory. config.json holds
the effective parameters, run status, labels, partition sizes and compact split
settings. Normalization and label mappings needed for inference stay in model
checkpoints. No function here deletes existing run artifacts or source data.
"""
from collections import Counter
import json
from pathlib import Path

from .artifacts import write_json


def update_run_config(run, **updates):
    """Update the existing parameter file instead of creating status sidecars."""
    path = Path(run) / "config.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    data.update(updates)
    write_json(path, data)


def compact_metadata(value):
    """Keep split settings/counts, without per-record or per-trial export lists."""
    omitted = {"selected_trials", "record_ids", "excluded_records", "records",
               "issues", "manifest", "normalization"}
    if isinstance(value, dict):
        return {key: compact_metadata(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, (list, tuple)):
        return [compact_metadata(item) for item in value]
    return value


def preparation_summary(folds, issues=(), label_mapping=None):
    """One small entry per fold; never serialize signals, intervals or samples."""
    summaries = []
    for fold in folds:
        summaries.append({
            "name": fold["name"],
            "split_summary": compact_metadata(fold.get("split_summary", fold.get("summary", {}))),
            "protocol": compact_metadata(fold.get("protocol_metadata", fold.get("metadata", {}))),
        })
    summary = {"fold_count": len(summaries), "folds": summaries,
               "warning_counts": dict(sorted(Counter(
                   issue.get("code", "unknown") for issue in issues
                   if issue.get("severity") == "warning").items()))}
    if label_mapping is not None:
        summary["label_mapping"] = label_mapping
    return summary


def save_run_error(run, error):
    """Save actionable failure details only; never export a recording manifest."""
    update_run_config(run, status="failed")
    details = {"error_type": type(error).__name__, "message": str(error)}
    issues = getattr(error, "issues", None)
    if issues is None and isinstance(getattr(error, "result", None), dict):
        issues = error.result.get("issues")
    if issues:
        details["issues"] = [item for item in issues if item.get("severity") == "error"]
    write_json(Path(run) / "error.json", details)

