"""Choose protocol1 checkpoints with equal weight for each validation person.

Compute macro F1 separately with the same full target vocabulary for every
validation participant, then take their arithmetic mean. Gesture-object tasks
use the joint classes. Other protocols keep pooled macro F1. Pooled loss still
breaks checkpoint ties and controls any loss-based scheduler. No test data is
accepted here. The original pooled metrics are retained in epoch history.
"""
from collections import defaultdict
from statistics import mean

from scripts.metrics import classification_metrics


def validation_selection(fold, has_validation):
    if not has_validation:
        return "last_epoch_no_validation"
    metadata = fold.get("protocol_metadata", {})
    kind = metadata.get("type", metadata.get("protocol_type"))
    if kind in ("leave_one_subject_out", "subject_holdout"):
        return "validation_subject_mean_macro_f1"
    return "validation_macro_f1"


def validation_score(metrics, targets, predictions, metadata, label_mapping, selection):
    """Return the monitored F1; reject missing/misaligned participant metadata."""
    if selection != "validation_subject_mean_macro_f1":
        return metrics["macro_f1"]
    if not len(targets) or not len(targets) == len(predictions) == len(metadata):
        raise ValueError("Validation targets, predictions and subject metadata must align")
    grouped = defaultdict(lambda: ([], []))
    for target, prediction, sample in zip(targets, predictions, metadata):
        subject = sample.get("subject_id")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("Protocol1 validation requires subject_id for every sample")
        grouped[subject][0].append(target)
        grouped[subject][1].append(prediction)
    return mean(classification_metrics(actual, predicted, label_mapping)["macro_f1"]
                for actual, predicted in grouped.values())


def print_validation_selection(selection):
    if selection == "validation_subject_mean_macro_f1":
        print("Validation F1: mean of per-participant macro F1 (equal participant weight); "
              "loss and accuracy remain pooled.", flush=True)
