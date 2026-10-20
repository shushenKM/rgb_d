"""Assign annotated trials to evaluation splits BEFORE generating windows.

PROTOCOL 1: UNSEEN SUBJECT
    make_subject_split holds out one entire subject. An optional DIFFERENT
    validation subject can be supplied. iter_subject_splits creates all LOSO
    folds. No random window-level validation split is created automatically.

PROTOCOL 2: HELD-OUT REPETITIONS
    The caller supplies original repetition IDs and optional gesture-specific
    rules. The accompanying protocol2 configuration describes this dataset:
      ordinary gestures: train 1-4, test 5;
      insert/deinsert:   train 1-4 and 6-9, test 5 and 10 (two blocks of five);
      screw:             train whole trials 1-2, test whole trial 3.
    Original repetition IDs remain unchanged. Gestures from the same
    (subject, session/step, object, repetition) stay in one partition. Any
    unassigned repetitions remain explicitly visible in the 'excluded' list.

SCREW SEGMENTS
    Each long parent contains repeated screws. Consecutive non-overlapping
    fixed-duration excerpts are made by the IMU sampling stage. This module
    preserves original parent intervals. Assign parent trials first,
    then generate windows inside their split; never randomly distribute windows
    of the same original trial across training and testing. Whole-trial holdout
    keeps every child excerpt within its parent's partition.

RETURN VALUE AND CHECKS
    A result contains train, validation, test, excluded, metadata, summary,
    issues, and valid. Input records are not modified. Checks detect record/trial
    overlap, missing class coverage, and overlapping time intervals in different
    partitions of the same subject/session/sensor. Strict mode (the default)
    raises SplitValidationError on errors; its .result preserves the complete
    diagnostic result. strict=False is for REVIEW, not permission to train.
    Missing signal files, recording correspondence, and duplicated raw signals
    between different subjects must be checked later by the IMU data loader.

VALIDATION AND SIGNAL AVAILABILITY
    Validation is optional and explicit at this stage. With none assigned, the
    result includes a warning; the test set must not be used for model selection.
    Records with unavailable modalities remain present. Their handling will be
    explicit in each modality's data adapter, followed by another split check.

USAGE
    split = make_subject_split(records, test_subject_id='1Alice')
    split = make_split_from_config(records, protocol2_config)
    preview = make_split_from_config(records, protocol2_config, strict=False)

Only Python's standard library is used. There is no signal slicing, window
generation, training, file writing, or source-data modification in this module.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import math


PARTITIONS = ("train", "validation", "test", "excluded")


class SplitValidationError(ValueError):
    """An invalid split plus its complete diagnostic result for inspection."""
    def __init__(self, result):
        self.result = result
        codes = sorted({issue["code"] for issue in result["issues"] if issue["severity"] == "error"})
        super().__init__("Split is not ready for training: {}. Inspect .result or use strict=False for review.".format(", ".join(codes)))


def _records(records):
    records = list(records)
    if not records:
        raise ValueError("Cannot split an empty annotation collection.")
    seen = set()
    for record in records:
        for field in ("record_id", "subject_id", "session_id", "gesture_label", "object_id"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError("Missing/invalid {} in an annotation record.".format(field))
        if type(record.get("repetition")) is not int or record["repetition"] <= 0:
            raise ValueError("Every record must have a positive integer repetition.")
        if record["record_id"] in seen:
            raise ValueError("Duplicate record ID: {}".format(record["record_id"]))
        seen.add(record["record_id"])
    return records


def _trial(record):
    return (record["subject_id"], record["session_id"], record["object_id"], record["repetition"])


def validate_split(result, *, allow_train_validation_overlap=False):
    """Return partition and temporal-overlap issues without changing any records."""
    if type(allow_train_validation_overlap) is not bool:
        raise ValueError("allow_train_validation_overlap must be a boolean.")
    issues = []
    def add(code, message, severity="error", **details):
        issues.append(dict(details, code=code, severity=severity, message=message))
    source_labels = {record["gesture_label"] for part in PARTITIONS for record in result[part]}
    seen_ids = {}
    seen_trials = {}
    interval_groups = defaultdict(list)
    for part in PARTITIONS:
        for record in result[part]:
            record_id = record["record_id"]
            if record_id in seen_ids:
                add("record_overlap", "A source row occurs more than once in the split.", record_id=record_id)
            seen_ids[record_id] = part
            if part == "excluded":
                continue
            key = _trial(record)
            if key in seen_trials and seen_trials[key] != part:
                add("trial_overlap", "One subject/session/object/repetition crosses partitions.", record_id=record_id)
            seen_trials[key] = part
            for timeline, interval in record.get("intervals", {}).items():
                start, end = interval.get("start_s"), interval.get("end_s")
                if (start is None or end is None or not math.isfinite(start)
                        or not math.isfinite(end) or not 0 <= start < end):
                    continue
                interval_groups[(record["subject_id"], record["session_id"], timeline)].append((start, end, part, record_id))
        labels = {record["gesture_label"] for record in result[part]}
        if part in ("train", "test"):
            if not result[part]:
                add("empty_partition", "{} is empty.".format(part), partition=part)
            missing = sorted(source_labels.difference(labels))
            if missing:
                add("missing_class_coverage", "{} lacks classes: {}.".format(part, ", ".join(missing)), partition=part)
        elif part == "validation" and result[part] and source_labels.difference(labels):
            add("validation_class_coverage", "Validation does not cover every class.", "warning")
    if not result["validation"]:
        add("no_validation_split", "No validation split was assigned; specify it before model selection.", "warning")
    if result["excluded"]:
        add("excluded_repetitions", "{} records are explicitly unassigned to train/validation/test.".format(len(result["excluded"])), "warning")

    for (subject, session, timeline), intervals in sorted(interval_groups.items()):
        active = {}
        for start, end, part, record_id in sorted(intervals):
            for other, (other_end, other_id) in active.items():
                if part != other and start < other_end:
                    allowed = allow_train_validation_overlap and {part, other} == {"train", "validation"}
                    add("overlapping_intervals", "Source time intervals overlap across {} and {}.".format(other, part),
                        severity="warning" if allowed else "error",
                        partitions=[other, part], allowed_by_configuration=allowed,
                        subject_id=subject, session_id=session, timeline=timeline,
                        record_id=record_id, related_record_id=other_id)
            if part not in active or end > active[part][0]:
                active[part] = (end, record_id)
    return issues


def _finish(result, strict):
    result["summary"] = {
        part: {
            "segments": len(result[part]),
            "subjects": sorted({record["subject_id"] for record in result[part]}),
            "gesture_counts": dict(sorted(Counter(record["gesture_label"] for record in result[part]).items())),
        } for part in PARTITIONS
    }
    result["issues"] = validate_split(result)
    result["valid"] = not any(issue["severity"] == "error" for issue in result["issues"])
    if strict and not result["valid"]:
        raise SplitValidationError(result)
    return result


def make_subject_split(records, test_subject_id, validation_subject_id=None, strict=True):
    """Protocol 1: hold out one subject and optionally a second for validation."""
    records = _records(records)
    subjects = {record["subject_id"] for record in records}
    if test_subject_id not in subjects:
        raise ValueError("Unknown or unspecified test subject: {!r}".format(test_subject_id))
    if validation_subject_id is not None:
        if validation_subject_id not in subjects or validation_subject_id == test_subject_id:
            raise ValueError("Validation subject must exist and differ from the test subject.")
    result = {part: [] for part in PARTITIONS}
    result["metadata"] = {"protocol": "protocol1", "type": "leave_one_subject_out",
                          "test_subject_id": test_subject_id, "validation_subject_id": validation_subject_id}
    for record in records:
        subject = record["subject_id"]
        part = "test" if subject == test_subject_id else "validation" if subject == validation_subject_id else "train"
        result[part].append(record)
    return _finish(result, strict)


def iter_subject_splits(records, validation_subjects=None, strict=True):
    """Yield all LOSO folds; optional mapping supplies a validation subject per fold."""
    records = _records(records)
    for subject in sorted({record["subject_id"] for record in records}):
        validation = None if validation_subjects is None else validation_subjects.get(subject)
        yield make_subject_split(records, subject, validation, strict=strict)


def _repetitions(values):
    values = [] if values is None else list(values)
    if any(type(value) is not int or value <= 0 for value in values) or len(values) != len(set(values)):
        raise ValueError("Repetition IDs must be unique positive integers.")
    return set(values)


def _rule(train, test, validation):
    rule = {"train": _repetitions(train), "test": _repetitions(test), "validation": _repetitions(validation)}
    if not rule["train"] or not rule["test"]:
        raise ValueError("A repetition rule requires nonempty training and testing IDs.")
    if any(rule[a].intersection(rule[b]) for a, b in (("train", "test"), ("train", "validation"), ("validation", "test"))):
        raise ValueError("Training, validation, and test repetition IDs must be disjoint.")
    return rule


def make_repetition_split(records, train_repetitions, test_repetitions,
                          validation_repetitions=None, gesture_rules=None, strict=True):
    """Protocol 2: apply explicit repetition rules and preserve original trial IDs.

    gesture_rules may override train_repetitions, test_repetitions, and
    validation_repetitions for individual gestures. Trial-group validation
    prevents inconsistent overrides from dividing one trial across partitions.
    """
    records = _records(records)
    base = _rule(train_repetitions, test_repetitions, validation_repetitions)
    overrides = {}
    for gesture, settings in (gesture_rules or {}).items():
        if gesture not in {record["gesture_label"] for record in records}:
            raise ValueError("Gesture rule names an unknown gesture: {}".format(gesture))
        overrides[gesture] = _rule(
            settings.get("train_repetitions", sorted(base["train"])),
            settings.get("test_repetitions", sorted(base["test"])),
            settings.get("validation_repetitions", sorted(base["validation"])),
        )
    result = {part: [] for part in PARTITIONS}
    result["metadata"] = {
        "protocol": "protocol2", "type": "held_out_repetitions",
        "default_rule": {part: sorted(values) for part, values in base.items()},
        "gesture_rules": {gesture: {part: sorted(values) for part, values in rule.items()} for gesture, rule in overrides.items()},
    }
    for record in records:
        rule = overrides.get(record["gesture_label"], base)
        part = next((name for name in ("train", "validation", "test") if record["repetition"] in rule[name]), "excluded")
        result[part].append(record)
    return _finish(result, strict)


def make_split_from_config(records, config, test_subject_id=None,
                           validation_subject_id=None, strict=True):
    """Dispatch a loaded protocol JSON configuration to its explicit split rule."""
    if config.get("split_before_windowing") is not True:
        raise ValueError("Protocols must assign source trials before windowing.")
    if config.get("validation") is not None:
        raise ValueError("Configure validation through explicit subject/repetition IDs.")
    if config.get("type") == "leave_one_subject_out":
        return make_subject_split(
            records, test_subject_id if test_subject_id is not None else config.get("test_subject_id"),
            validation_subject_id if validation_subject_id is not None else config.get("validation_subject_id"), strict=strict,
        )
    if config.get("type") == "held_out_repetitions":
        result = make_repetition_split(
            records, config["train_repetitions"], config["test_repetitions"],
            config.get("validation_repetitions"), config.get("gesture_rules"), strict=False,
        )
        for key in ("repetition_blocks", "screw_sampling_plan"):
            if key in config:
                result["metadata"][key] = deepcopy(config[key])
        if strict and not result["valid"]:
            raise SplitValidationError(result)
        return result
    raise ValueError("Unknown protocol type: {!r}".format(config.get("type")))
