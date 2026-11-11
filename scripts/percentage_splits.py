"""Percentage splits of complete micro-gesture source trials.

Split before windows or screw excerpts. Trial keys include subject, session,
object and original repetition. Trials with the same set of target classes are
shuffled together, then allocated by largest remainder to the requested ratios.
This retains class coverage and makes splits reproducible across modalities and
input row order. Ratios target trial counts; annotation/window ratios can differ.
"""
from collections import defaultdict
import math
import random

from scripts.protocol_config import percentage_settings
from scripts.protocols import PARTITIONS, _records
from scripts.tasks import label_of

ACTIVE_PARTS = ("train", "validation", "test")


def _counts(size, percentages):
    active = [p for p in ACTIVE_PARTS if percentages[p] > 0]
    quotas = {p: size * percentages[p] / 100 for p in ACTIVE_PARTS}
    minimum = 1 if size >= len(active) else 0
    counts = {p: max(minimum if p in active else 0, math.floor(quotas[p])) for p in ACTIVE_PARTS}
    while sum(counts.values()) < size:
        part = max(active, key=lambda p: quotas[p] - counts[p])
        counts[part] += 1
    while sum(counts.values()) > size:
        part = max((p for p in active if counts[p] > minimum), key=lambda p: counts[p] - quotas[p])
        counts[part] -= 1
    return counts


def make_percentage_split(records, protocol, *, seed=42, allow_train_validation_overlap=False):
    """Return assignments; the training splitter performs the final overlap checks."""
    if protocol.get("split_before_windowing") is not True:
        raise ValueError("Protocols must assign source trials before windowing")
    records = _records(records)
    percentages, split_seed = percentage_settings(protocol, seed)
    allow_test = protocol.get("allow_within_subject_test_overlap", False)
    if type(allow_test) is not bool or type(allow_train_validation_overlap) is not bool:
        raise ValueError("Overlap settings must be booleans")
    groups = defaultdict(list)
    for row in records:
        key = tuple(row[k] for k in ("subject_id", "session_id", "object_id", "repetition"))
        groups[key].append(row)
    signatures = {key: tuple(sorted({label_of(r) for r in rows})) for key, rows in groups.items()}
    labels = sorted({label for signature in signatures.values() for label in signature})
    active_count = sum(percentages[p] > 0 for p in ACTIVE_PARTS)
    scarce = [label for label in labels if sum(label in s for s in signatures.values()) < active_count]
    if scarce:
        raise ValueError("Percentage class coverage needs at least {} distinct complete trials per class: {}".format(
            active_count, ", ".join(scarce)))
    strata = defaultdict(list)
    for key, signature in signatures.items():
        strata[signature].append(key)
    rng = random.Random(split_seed)
    assignments = {}
    counts = dict.fromkeys(ACTIVE_PARTS, 0)
    for signature in sorted(strata):
        keys = sorted(strata[signature])
        rng.shuffle(keys)
        allocated = _counts(len(keys), percentages)
        cursor = 0
        for part in ACTIVE_PARTS:
            for key in keys[cursor:cursor + allocated[part]]:
                assignments[key] = part
            counts[part] += allocated[part]
            cursor += allocated[part]
    result = {part: [] for part in PARTITIONS}
    for row in records:
        key = tuple(row[k] for k in ("subject_id", "session_id", "object_id", "repetition"))
        result[assignments[key]].append(row)
    result["name"] = protocol.get("name", "protocol2")
    result["metadata"] = dict(
        protocol=result["name"], type="grouped_percentages", grouped_percentages=True,
        allow_within_subject_test_overlap=allow_test,
        validation=dict(strategy="percentage_partition",
                        allow_train_validation_overlap=allow_train_validation_overlap),
        percentage_split=dict(
            requested_percentages=percentages, split_seed=split_seed,
            unit="whole_subject_session_object_repetition_trial",
            stratification="target_class_composition",
            allocation="largest_remainder_minimum_one_per_nonzero_partition_when_possible",
            trial_counts=counts, total_trials=len(groups),
            annotation_counts={p: len(result[p]) for p in ACTIVE_PARTS},
            actual_trial_percentages={p: 100 * counts[p] / len(groups) for p in ACTIVE_PARTS},
            actual_annotation_percentages={p: 100 * len(result[p]) / len(records) for p in ACTIVE_PARTS}))
    return result


def percentage_split_summary(metadata):
    """One terminal line with requested and realized sizes; no extra output files."""
    summary = metadata.get("percentage_split")
    if not summary:
        return None
    parts = ("train", "validation", "test")
    return ("train/validation/test requested {}/{}/{}% | actual trials {:.2f}/{:.2f}/{:.2f}% "
            "| actual annotations {:.2f}/{:.2f}/{:.2f}%").format(
        *(format(summary["requested_percentages"][p], "g") for p in parts),
        *(summary["actual_trial_percentages"][p] for p in parts),
        *(summary["actual_annotation_percentages"][p] for p in parts))
