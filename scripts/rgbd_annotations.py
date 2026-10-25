"""Read RGB-D gesture CSV annotations without changing any source files.

The EN export uses commas and decimal points. Its original identifiers match
the IMU/sEMG annotation schema, allowing the same labels and evaluation split
helpers to be reused. Above uses start_time/end_time; front uses
start_time2/end_time2; left uses start_time4/end_time4; right uses
start_time3/end_time3. These times refer to each camera's VIDEO PLAYBACK clock.
Reference clocks are retained as metadata and are never added as offsets.

Separator rows are skipped, identifiers are stripped in memory, and malformed
rows remain visible through parse_errors. Row IDs retain the source CSV line.
No annotations are repaired, rewritten, inferred from another view, or saved.
"""

import csv
import math
from pathlib import Path


VIEW_COLUMNS = {
    "above": ("start_time", "end_time", "rgb_above_reference_time"),
    "front": ("start_time2", "end_time2", "rgb_front_reference_time"),
    "left": ("start_time4", "end_time4", "rgb_left_reference_time"),
    "right": ("start_time3", "end_time3", "rgb_right_reference_time"),
}
IDENTIFIERS = ("session_id", "gesture_label", "object", "repetition")


def _text(value):
    return "" if value is None else str(value).strip()


def _number(value, field, errors):
    value = _text(value)
    if not value:
        return None
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError()
        return result
    except ValueError:
        errors.append({"code": "invalid_annotation_number", "field": field, "value": value})
        return None


def read_rgbd_annotations(path, subject_id):
    """Return provenance-preserving parent records from one EN CSV export."""
    path = Path(path)
    subject_id = _text(subject_id)
    if not subject_id:
        raise ValueError("subject_id must be nonempty")
    if path.suffix.lower() != ".csv":
        raise ValueError("RGB-D annotations must be EN CSV files")
    source = "/".join((subject_id, path.parent.name, path.name))
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=",")
        headers = [_text(value) for value in next(reader, [])]
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate annotation column names in {}".format(source))
        required = set(IDENTIFIERS)
        required.update(c for columns in VIEW_COLUMNS.values() for c in columns[:2])
        missing = sorted(required.difference(headers))
        if missing:
            raise ValueError("Missing RGB-D columns in {}: {}".format(source, ", ".join(missing)))
        for cells in reader:
            line = reader.line_num
            row = dict(zip(headers, cells))
            relevant = ["gesture_label", "repetition"] + [c for columns in VIEW_COLUMNS.values() for c in columns[:2]]
            if not any(_text(row.get(key)) for key in relevant):
                continue
            errors = []
            if len(cells) != len(headers):
                errors.append({"code": "annotation_row_width", "expected": len(headers), "actual": len(cells)})
            repetition = _number(row.get("repetition"), "repetition", errors)
            if repetition is not None and (repetition <= 0 or repetition != int(repetition)):
                errors.append({"code": "invalid_repetition", "field": "repetition", "value": repetition})
                repetition = None
            intervals = {}
            for view, (start, end, reference) in VIEW_COLUMNS.items():
                intervals[view] = {
                    "start_s": _number(row.get(start), start, errors),
                    "end_s": _number(row.get(end), end, errors),
                    "reference_time": _text(row.get(reference)),
                }
            records.append({
                "record_id": "{}:{}".format(source, line),
                "subject_id": subject_id,
                "session_id": _text(row.get("session_id")),
                "gesture_label": _text(row.get("gesture_label")),
                "object_id": _text(row.get("object")),
                "repetition": None if repetition is None else int(repetition),
                "source_file": source,
                "source_row": line,
                "intervals": intervals,
                "parse_errors": errors,
            })
    return records


def read_dataset_rgbd_annotations(dataset_root, dataset_config, subject_ids=None):
    """Read exactly one selected EN export per subject, with no file writes."""
    root = Path(dataset_root).resolve()
    if dataset_config.get("annotation_locale", "EN") != "EN":
        raise ValueError("RGB-D training currently requires annotation_locale='EN'")
    selected = subject_ids if subject_ids is not None else dataset_config.get("subject_ids")
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir()) if selected is None else list(selected)
    if (not subjects or len(set(subjects)) != len(subjects)
            or any(not isinstance(s, str) or not s.strip() or Path(s).name != s for s in subjects)):
        raise ValueError("subject_ids must contain unique subject directory names")
    folder = dataset_config.get("folders", {}).get("annotations", "annotation")
    pattern = dataset_config.get("annotation_file_pattern") or "rgb_depth_gesture_segments*_EN.csv"
    records = []
    for subject in subjects:
        directory = (root / subject / folder).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Annotation directory must stay inside dataset_root")
        paths = sorted(directory.glob(pattern))
        if len(paths) != 1:
            raise ValueError("Expected exactly one {} in {}; found {}".format(pattern, directory, len(paths)))
        if not paths[0].resolve().is_relative_to(root):
            raise ValueError("Annotation file must stay inside dataset_root")
        records.extend(read_rgbd_annotations(paths[0], subject))
    return records
