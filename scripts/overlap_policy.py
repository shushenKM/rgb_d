"""Apply the configured policy for signal reuse across data partitions.

Both modality configs may enable data_quality.allow_cross_subject_overlap.
When enabled, matching signals under different participant IDs are reported as
warnings, including matches involving test data. Same-participant overlaps
still follow the existing training/validation policy. This module does not
change records, assign splits, or read or write any dataset files.
"""


def cross_subject_overlap_enabled(data_quality):
    """Read an explicit boolean; omitted settings retain strict checks."""
    enabled = data_quality.get("allow_cross_subject_overlap", False)
    if type(enabled) is not bool:
        raise ValueError("data_quality.allow_cross_subject_overlap must be a boolean.")
    return enabled


def allows_overlap(split, left, right, left_record, right_record, *,
                   allow_cross_subject_overlap=False):
    """Return whether detected cross-partition reuse should be a warning."""
    if left == right:
        return False
    if allow_cross_subject_overlap and left_record["subject_id"] != right_record["subject_id"]:
        return True
    return ({left, right} == {"train", "validation"}
            and split.get("metadata", {}).get("validation", {}).get("allow_train_validation_overlap", False))
