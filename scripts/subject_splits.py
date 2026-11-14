"""Configure participant-disjoint protocol1 folds before reading signal data.

One test participant with no fixed IDs runs leave-one-subject-out: every selected
participant is tested once with a fresh model. A fixed test_subject_ids list runs
one fold, including when it contains two participants. subject_counts controls
train/validation/test participant counts; null training means all remaining.
Null validation uses validation_subject_fraction (default 0.25) of all selected
participants, rounded to nearest integer with halves rounded up and at least
one participant. Fixed validation IDs override this automatic count.

Automatic validation takes the next available participants in cyclic sorted
order after the first test participant. Training takes the remaining participants
in that order, up to the requested count; unused participants are excluded from
every context. This deterministic assignment is shared by all modalities and
experiments. Legacy singular test/validation settings remain supported.
"""
from copy import deepcopy
import math


SUBJECT_PROTOCOL_TYPES = ("subject_holdout", "leave_one_subject_out")


def _subject_list(value, name):
    if value is None:
        return None
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item.strip() or item != item.strip()
                   or item in (".", "..") or "/" in item or "\\" in item for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(name + " must be a nonempty list of unique participant folder names, or null")
    return sorted(value)


def normalize_subject_protocol(protocol):
    """Validate settings without data access; return a copy with canonical keys."""
    result = deepcopy(protocol)
    if result.get("type") not in SUBJECT_PROTOCOL_TYPES:
        raise ValueError("Subject holdout settings apply only to protocol1")
    legacy = result.get("validation") or {}
    if not isinstance(legacy, dict) or legacy.get("strategy", "auto") not in ("auto", "next_subject", "none"):
        raise ValueError("Participant validation must use next_subject or none")
    supplied = result.get("subject_counts", {})
    if not isinstance(supplied, dict) or set(supplied) - {"train", "validation", "test"}:
        raise ValueError("subject_counts accepts train, validation and test")
    ids = {}
    for part in ("test", "validation"):
        candidates = [result.get(part + "_subject_ids")]
        singulars = [result.get(part + "_subject_id")]
        if part == "validation":
            singulars.append(legacy.get("subject_id"))
        candidates += [[value] for value in singulars if value is not None]
        lists = [_subject_list(value, part + "_subject_ids") for value in candidates if value is not None]
        if lists and any(value != lists[0] for value in lists):
            raise ValueError("Conflicting singular/plural " + part + " participant settings")
        ids[part] = lists[0] if lists else None
    fraction = result.get("validation_subject_fraction", 0.25)
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("validation_subject_fraction must be strictly between zero and one")
    result["validation_subject_fraction"] = fraction
    counts = {"train": supplied.get("train"),
              "validation": supplied.get("validation"),
              "test": supplied.get("test", len(ids["test"] or [None]))}
    if counts["validation"] is None:
        if legacy.get("strategy") == "none":
            counts["validation"] = 0
        elif ids["validation"] is not None:
            counts["validation"] = len(ids["validation"])
    for part, count in counts.items():
        if part in ("train", "validation") and count is None:
            continue
        if type(count) is not int or count < (0 if part == "validation" else 1):
            raise ValueError("subject_counts.{} must be {}".format(part,
                "a nonnegative integer or null" if part == "validation" else "a positive integer" + (" or null" if part == "train" else "")))
    if legacy.get("strategy") == "none" and counts["validation"] != 0:
        raise ValueError("validation.strategy=none conflicts with a nonzero validation subject count")
    for part in ids:
        if ids[part] is not None and len(ids[part]) != counts[part]:
            raise ValueError("{}_subject_ids length must match subject_counts.{}".format(part, part))
    if ids["test"] is None:
        if counts["test"] != 1:
            raise ValueError("Specify test_subject_ids (or --test-subjects) when holding out two or more test participants")
        if ids["validation"] is not None:
            raise ValueError("Fixed validation participants require fixed test_subject_ids; otherwise they would also become a test fold")
    elif set(ids["test"]) & set(ids["validation"] or []):
        raise ValueError("Validation and test participants must be disjoint")
    result["subject_counts"] = counts
    result.update({part + "_subject_ids": value for part, value in ids.items()})
    result["subject_selection"] = "fixed" if ids["test"] else "leave_one_subject_out"
    return result


def participant_partitions(subjects, protocol):
    """Return one fixed partition or one fold per available test participant."""
    settings = normalize_subject_protocol(protocol)
    subjects = sorted(set(subjects))
    counts = dict(settings["subject_counts"])
    if counts["validation"] is None:
        counts["validation"] = max(1, math.floor(len(subjects) * settings["validation_subject_fraction"] + 0.5))
    if len(subjects) < 3 and counts["validation"] > 0:
        raise ValueError("Participant-disjoint validation requires at least three subjects")
    train_count = counts["train"]
    if train_count is None:
        train_count = len(subjects) - counts["validation"] - counts["test"]
    if train_count < 1 or train_count + counts["validation"] + counts["test"] > len(subjects):
        raise ValueError("Requested train/validation/test subject counts exceed the {} available participants, or leave no training participant".format(len(subjects)))
    for part in ("test", "validation"):
        unknown = set(settings[part + "_subject_ids"] or []) - set(subjects)
        if unknown:
            raise ValueError("Unknown {} subject/participant(s): {}".format(part, ", ".join(sorted(unknown))))
    test_sets = [settings["test_subject_ids"]] if settings["test_subject_ids"] else [[subject] for subject in subjects]
    partitions = []
    for test in test_sets:
        start = subjects.index(test[0]) + 1
        order = subjects[start:] + subjects[:start]
        candidates = [subject for subject in order if subject not in test]
        validation = settings["validation_subject_ids"]
        if validation is None:
            validation = candidates[:counts["validation"]]
        train = [subject for subject in candidates if subject not in validation][:train_count]
        unused = sorted(set(subjects) - set(train) - set(validation) - set(test))
        partitions.append(dict(train=sorted(train), validation=sorted(validation), test=sorted(test),
                               excluded=unused, selection=settings["subject_selection"],
                               requested_counts=dict(settings["subject_counts"])))
    return partitions


def participant_split_summary(metadata):
    """Compact participant assignments for the preparation terminal output."""
    if "subject_counts" not in metadata:
        return None
    parts = ("train", "validation", "test")
    return "participants train/validation/test {}/{}/{} | train [{}] | validation [{}] | test [{}]".format(
        *(metadata["subject_counts"][p] for p in parts),
        *(", ".join(metadata[p + "_subject_ids"]) or "none" for p in parts))
