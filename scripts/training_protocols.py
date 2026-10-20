"""Build validation-aware, source-trial splits before any signal windowing.

The original protocol implementations remain the authority for held-out test
subjects/repetitions and leakage checks. This module only selects validation
data from their training partition. It never reads or modifies source files.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import math
import random

from scripts.protocols import (
    PARTITIONS,
    SplitValidationError,
    make_split_from_config,
    validate_split,
)


def _trial(record):
    return (record["subject_id"], record["session_id"], record["object_id"], record["repetition"])


def _finish(result, allow_train_validation_overlap=False):
    """Refresh summaries/checks after moving complete trials into validation."""
    result["summary"] = {
        part: {
            "segments": len(result[part]),
            "subjects": sorted({record["subject_id"] for record in result[part]}),
            "gesture_counts": dict(sorted(Counter(record["gesture_label"] for record in result[part]).items())),
        }
        for part in PARTITIONS
    }
    result["issues"] = validate_split(result, allow_train_validation_overlap=allow_train_validation_overlap)
    labels = {record["gesture_label"] for part in PARTITIONS for record in result[part]}
    if result["validation"]:
        missing = sorted(labels.difference(record["gesture_label"] for record in result["validation"]))
        if missing:
            result["issues"].append({
                "code": "missing_validation_class_coverage", "severity": "error",
                "message": "Validation lacks classes: {}.".format(", ".join(missing)),
                "partition": "validation", "missing_classes": missing,
            })
    result["valid"] = not any(issue["severity"] == "error" for issue in result["issues"])
    if not result["valid"]:
        raise SplitValidationError(result)
    return result


def _validation_settings(protocol_config, validation_config, seed):
    settings = validation_config
    if settings is None:
        settings = protocol_config.get("validation")
    if settings is None:
        settings = {"strategy": "auto"}
    if not isinstance(settings, dict):
        raise ValueError("Validation configuration must be an object with a strategy.")
    settings = deepcopy(settings)
    strategy = settings.get("strategy", "auto")
    if strategy == "auto":
        strategy = "next_subject" if protocol_config.get("type") == "leave_one_subject_out" else "grouped_trials"
    settings["strategy"] = strategy
    settings.setdefault("seed", seed)
    if type(settings["seed"]) is not int:
        raise ValueError("Validation seed must be an integer.")
    settings.setdefault("allow_train_validation_overlap", False)
    if type(settings["allow_train_validation_overlap"]) is not bool:
        raise ValueError("allow_train_validation_overlap must be a boolean.")
    return settings


def _select_validation_groups(groups, labels, fraction, seed):
    """Select whole groups with every class retained on both sides.

    First cover every validation class using a deterministic backtracking search
    (at most one addition per newly covered class), then fill toward the target
    group fraction while retaining training coverage. The requested fraction is
    a target: class coverage takes priority and the actual fraction is recorded.
    """
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Grouped validation needs at least two distinct training trials.")
    classes = {key: {record["gesture_label"] for record in groups[key]} for key in keys}
    label_groups = {label: {key for key in keys if label in classes[key]} for label in labels}
    rare = sorted(label for label, choices in label_groups.items() if len(choices) < 2)
    if rare:
        raise ValueError(
            "Cannot create grouped validation while retaining every class in training: "
            "these classes need at least two distinct training trials: {}. "
            "Add eligible trials or explicitly disable validation with strategy='none'.".format(", ".join(rare))
        )
    order = list(keys)
    random.Random(seed).shuffle(order)
    priority = {key: index for index, key in enumerate(order)}
    target = max(1, min(len(keys) - 1, int(math.floor(len(keys) * fraction + 0.5))))
    visited = set()

    def viable(selected, key):
        proposed = selected | {key}
        return all(not choices.issubset(proposed) for choices in label_groups.values())

    def cover(selected):
        state = frozenset(selected)
        if state in visited:
            return None
        visited.add(state)
        covered = set().union(*(classes[key] for key in selected)) if selected else set()
        missing = set(labels).difference(covered)
        if not missing:
            return selected
        choices_by_label = {
            label: [key for key in label_groups[label] if key not in selected and viable(selected, key)]
            for label in missing
        }
        label = min(missing, key=lambda item: (len(choices_by_label[item]), item))
        choices = sorted(choices_by_label[label], key=lambda key: (-len(classes[key].intersection(missing)), priority[key]))
        for key in choices:
            found = cover(selected | {key})
            if found is not None:
                return found
        return None

    selected = cover(set())
    if selected is None:
        raise ValueError(
            "Cannot assign complete training trials to validation while keeping every class "
            "in both partitions. Review trial grouping or explicitly use validation strategy='none'."
        )
    for key in order:
        if len(selected) >= target:
            break
        if key not in selected and viable(selected, key):
            selected.add(key)
    return selected, target


def build_training_splits(records, protocol_config, validation_config=None, seed=42):
    """Return strict training/validation/test split dictionaries for a protocol.

    ``test_subject_id: null`` builds all LOSO folds. ``auto`` validation means
    the next sorted subject for LOSO, or seeded complete-trial selection for
    held-out repetitions. ``strategy: none`` explicitly disables validation.
    The original records and configuration are never changed. Pass only the
    relevant modality's interval timelines when checking modality-specific data.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot build training splits from empty records.")
    if not isinstance(protocol_config, dict):
        raise ValueError("Protocol configuration must be an object.")
    config = deepcopy(protocol_config)
    validation = _validation_settings(config, validation_config, seed)
    config["validation"] = None  # The low-level API accepts explicit IDs only.
    strategy = validation["strategy"]
    allow_overlap = validation["allow_train_validation_overlap"]
    protocol_type = config.get("type")

    if protocol_type == "leave_one_subject_out":
        if strategy not in ("next_subject", "none"):
            raise ValueError("LOSO validation strategy must be 'auto', 'next_subject', or 'none'.")
        subjects = sorted({record["subject_id"] for record in records})
        requested_test = config.get("test_subject_id")
        tests = subjects if requested_test is None else [requested_test]
        explicit_validation = validation.get("validation_subject_id", validation.get("subject_id", config.get("validation_subject_id")))
        if strategy == "none" and explicit_validation is not None:
            raise ValueError("Explicit validation_subject_id conflicts with validation strategy='none'.")
        if strategy != "none" and len(subjects) < 3:
            raise ValueError("LOSO training with a separate validation subject requires at least three subjects.")
        results = []
        for test in tests:
            if test not in subjects:
                raise ValueError("Unknown test subject: {!r}".format(test))
            held_validation = None
            if strategy != "none":
                held_validation = explicit_validation
                if held_validation is None:
                    held_validation = subjects[(subjects.index(test) + 1) % len(subjects)]
            fold_config = dict(config, validation_subject_id=held_validation)
            result = make_split_from_config(records, fold_config, test_subject_id=test, strict=False)
            result["name"] = "{}_{}".format(config.get("name", "protocol1"), test)
            result["metadata"]["validation"] = dict(validation, validation_subject_id=held_validation)
            results.append(_finish(result, allow_overlap))
        return results

    if protocol_type == "held_out_repetitions":
        if strategy not in ("grouped_trials", "none"):
            raise ValueError("Repetition validation strategy must be 'auto', 'grouped_trials', or 'none'.")
        result = make_split_from_config(records, config, strict=False)
        result["name"] = config.get("name", "protocol2")
        _finish(result, allow_overlap)
        if result["validation"]:
            if strategy == "none":
                raise ValueError("Configured validation repetitions conflict with validation strategy='none'.")
            result["metadata"]["validation"] = dict(validation, strategy="explicit_repetitions")
            return [_finish(result, allow_overlap)]
        if strategy == "none":
            result["metadata"]["validation"] = dict(validation)
            return [_finish(result, allow_overlap)]
        fraction = validation.get("fraction", 0.2)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Grouped validation fraction must be finite and strictly between zero and one.")
        groups = defaultdict(list)
        for record in result["train"]:
            groups[_trial(record)].append(record)
        labels = {record["gesture_label"] for part in PARTITIONS for record in result[part]}
        selected, target = _select_validation_groups(groups, labels, fraction, validation["seed"])
        original_train = result["train"]
        result["validation"] = [record for record in original_train if _trial(record) in selected]
        result["train"] = [record for record in original_train if _trial(record) not in selected]
        result["metadata"]["validation"] = dict(
            validation, fraction=fraction, target_trial_count=target,
            selected_trial_count=len(selected), eligible_trial_count=len(groups),
            actual_fraction=len(selected) / len(groups),
            selected_trials=[list(key) for key in sorted(selected)],
        )
        return [_finish(result, allow_overlap)]

    raise ValueError("Unknown protocol type: {!r}".format(protocol_type))
