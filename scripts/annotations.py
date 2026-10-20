"""Read and check the annotations shared by IMU and sEMG training.

PURPOSE
    Read one configured EN or FR CSV export per subject and return one record
    per annotated gesture interval. IMU and sEMG training will use these same
    records, with their own time boundaries. This module does not read signals,
    assign class indices, split repetitions, generate windows, or train models.

INPUT AND PORTABLE PATHS
    Expected layout: Manipulative_dataset/{dataset,imu_semg}.
    The dataset is located relative to this script's repository, independently
    of the shell/IDE working directory. Settings come from configs/dataset.json.
    annotation_locale selects EN (comma delimiter, decimal point) or FR
    (semicolon delimiter, decimal comma), independently of the computer's
    language. Only the selected *_EN.csv or *_FR.csv files are loaded.
    Only Python's standard library is required.

COLUMN MAPPING
    IMU E45A: imu_start_s,   imu_end_s,   imu_45A_reference_time
    IMU E477: imu_start_s_1, imu_end_s_2, imu_477_reference_time
    sEMG:     semg_start_s,  semg_end_s,  semg_reference_time

PROCESSING RULES
    - Trim surrounding whitespace from identifiers and gesture labels.
    - Skip separator rows containing no gesture, repetition, or interval data.
    - Preserve subject, session/step, object, repetition, source file, CSV line,
      and the original cell values. Separate occurrences of 'move' stay separate.
    - Convert interval boundaries to seconds and repetition to a positive integer.
    - Keep missing/invalid values visible; never copy boundaries between sensors.
    - Carry sparse reference-clock cells forward only within the same subject,
      file, and session/step; retain the original cell and its source line.
    - Report malformed values, missing intervals, and duplicate time
      intervals. Do not repair annotations, apply offsets, or drop gesture rows.
    - Reference columns are optional metadata, never required for training.

OUTPUT
    read_annotations(path, subject_id) returns a list of dictionaries. Each has
    identifiers, intervals['imu_e45a'/'imu_e477'/'semg'], source provenance, and
    parsing errors. has_imu_pair means BOTH numeric intervals are valid; it does
    not certify clock alignment, signal quality, or recording correspondence.
    session_id is preserved as annotated (e.g. 'step1'); it is not assumed to be
    a recording filename. Reference clocks remain text until alignment work.

    validate_annotations(records) returns issues without changing the records.
    summarize_annotations(records, issues) returns counts and repetition groups.

USAGE FROM THE REPOSITORY ROOT
    python -m scripts.annotations                       # Console summary only
    python -m scripts.annotations --subjects 1Alice     # One subject
    python -m scripts.annotations --export             # Save a review report

DATA PROTECTION
    All source files are opened for reading only. Optional exports go into a NEW
    directory beneath runs/imu/annotations/, never into dataset/. They contain
    annotations.json, summary.json, and validation_issues.csv. Source data is
    never edited, renamed, moved, or deleted. Existing export files are not
    overwritten. The CLI exits with status 2 for errors; warnings remain visible.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMELINES = {
    "imu_e45a": ("imu_start_s", "imu_end_s", "imu_45A_reference_time"),
    "imu_e477": ("imu_start_s_1", "imu_end_s_2", "imu_477_reference_time"),
    "semg": ("semg_start_s", "semg_end_s", "semg_reference_time"),
}
IDENTIFIER_COLUMNS = ("session_id", "gesture_label", "object", "repetition")
REFERENCE_COLUMNS = tuple(columns[2] for columns in TIMELINES.values())
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + tuple(
    field for columns in TIMELINES.values() for field in columns[:2]
)
CLOCK_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:[.:,]\d{1,9})?$")
CSV_FORMATS = {"EN": {"delimiter": ",", "decimal": "."},
               "FR": {"delimiter": ";", "decimal": ","}}


def _project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _clean(value):
    return "" if value is None else str(value).strip()


def _annotation_locale(value):
    if not isinstance(value, str) or value.strip().upper() not in CSV_FORMATS:
        raise ValueError("annotation_locale must be EN or FR.")
    return value.strip().upper()


def _number(value, field, errors, annotation_locale="EN"):
    text = _clean(value)
    if not text:
        return None
    try:
        number = float(text.replace(CSV_FORMATS[annotation_locale]["decimal"], "."))
    except ValueError:
        errors.append({"code": "invalid_number", "field": field, "value": text})
        return None
    if not math.isfinite(number):
        errors.append({"code": "nonfinite_number", "field": field, "value": text})
        return None
    return number


def _valid_interval(interval):
    start, end = interval["start_s"], interval["end_s"]
    return start is not None and end is not None and 0 <= start < end


def read_annotations(annotation_path, subject_id, annotation_locale="EN"):
    """Read one CSV without writes; retain malformed gesture rows for validation.

    The selected locale controls the delimiter and numeric decimal separator.
    Numeric values and sparse reference clocks are normalized in memory only.
    The source filename is stored relative to the subject's annotation folder,
    so record IDs remain stable when the parent directory changes computers.
    """
    annotation_locale = _annotation_locale(annotation_locale)
    path = _project_path(annotation_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Use an EN or FR CSV annotation export: {}".format(path.name))
    subject_id = _clean(subject_id)
    if not subject_id:
        raise ValueError("subject_id must not be empty.")
    source_file = "/".join((subject_id, path.parent.name, path.name))
    records = []
    references = {}
    active_session = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=CSV_FORMATS[annotation_locale]["delimiter"])
        headers = [_clean(cell) for cell in next(reader, [])]
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate CSV column names in {}".format(source_file))
        missing = sorted(set(REQUIRED_COLUMNS).difference(headers))
        if missing:
            raise ValueError("Missing columns in {}: {}".format(source_file, ", ".join(missing)))
        while True:
            source_row = reader.line_num + 1
            try:
                cells = next(reader)
            except StopIteration:
                break
            if not cells:
                continue
            row = dict(zip(headers, cells))
            session_id = _clean(row.get("session_id"))
            if session_id and session_id != active_session:
                references = {}
                active_session = session_id
            if session_id:
                for name, columns in TIMELINES.items():
                    reference = _clean(row.get(columns[2]))
                    if reference:
                        references[name] = (reference, source_row)
            interval_columns = [field for columns in TIMELINES.values() for field in columns[:2]]
            if not any(_clean(row.get(field)) for field in
                       ["gesture_label", "repetition"] + interval_columns):
                continue

            errors = []
            if len(cells) > len(headers):
                errors.append({"code": "extra_cells", "field": "CSV row", "value": str(cells[len(headers):])})
            repetition = _number(row.get("repetition"), "repetition", errors, annotation_locale)
            if repetition is not None:
                if repetition <= 0 or not repetition.is_integer():
                    errors.append({"code": "invalid_repetition", "field": "repetition", "value": _clean(row.get("repetition"))})
                    repetition = None
                else:
                    repetition = int(repetition)
            intervals = {}
            for name, (start_column, end_column, reference_column) in TIMELINES.items():
                raw_reference = _clean(row.get(reference_column))
                reference, reference_row = references.get(name, (None, None)) if session_id else (raw_reference or None, source_row if raw_reference else None)
                interval = {
                    "start_s": _number(row.get(start_column), start_column, errors, annotation_locale),
                    "end_s": _number(row.get(end_column), end_column, errors, annotation_locale),
                    "reference_time": reference,
                    "reference_time_raw": raw_reference or None,
                    "reference_source_row": reference_row,
                    "reference_inherited": reference_row is not None and reference_row != source_row,
                }
                interval["valid_interval"] = _valid_interval(interval)
                intervals[name] = interval
            records.append({
                "record_id": "{}:row{}".format(source_file, source_row),
                "subject_id": subject_id,
                "session_id": session_id,
                "gesture_label": _clean(row.get("gesture_label")),
                "object_id": _clean(row.get("object")),
                "repetition": repetition,
                "source_file": source_file,
                "annotation_locale": annotation_locale,
                "source_row": source_row,
                "source_values": {field: row.get(field, "") for field in REQUIRED_COLUMNS + REFERENCE_COLUMNS + ("offset_s",)},
                "intervals": intervals,
                "has_imu_pair": all(intervals[name]["valid_interval"] for name in ("imu_e45a", "imu_e477")),
                "parse_errors": errors,
            })
    return records


def read_dataset_annotations(dataset_root=None, subject_ids=None,
                             annotation_pattern=None, annotation_folder="annotation",
                             annotation_locale="EN"):
    """Read exactly one matching CSV per selected subject, in stable order.

    Only files ending in the selected _EN.csv or _FR.csv suffix are considered,
    even with a broad custom pattern. Custom patterns may contain {locale}.
    Missing/ambiguous matches raise; the other locale is never a fallback.
    """
    annotation_locale = _annotation_locale(annotation_locale)
    annotation_pattern = (annotation_pattern or "*_{locale}.csv").replace("{locale}", annotation_locale)
    suffix = "_{}.csv".format(annotation_locale).lower()
    root = _project_path("../dataset" if dataset_root is None else dataset_root)
    if not root.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(root))
    if any(separator in annotation_folder for separator in ("/", "\\")) or annotation_folder in ("", ".", ".."):
        raise ValueError("annotation_folder must be a single folder name.")
    if not annotation_pattern or any(separator in annotation_pattern for separator in ("/", "\\")):
        raise ValueError("annotation_pattern must be a filename pattern.")
    if subject_ids is None:
        subjects = sorted(path.name for path in root.iterdir()
                          if path.is_dir() and (path / annotation_folder).is_dir())
    else:
        subjects = list(subject_ids)
        if len(subjects) != len(set(subjects)):
            raise ValueError("Subject selection contains duplicates.")
        subjects.sort()
    if not subjects:
        raise ValueError("No subject annotation folders were found or selected.")
    records = []
    for subject in subjects:
        if subject in ("", ".", "..") or any(separator in subject for separator in ("/", "\\")):
            raise ValueError("Each subject ID must be a folder name.")
        folder = root / subject / annotation_folder
        matches = sorted(path for path in folder.glob(annotation_pattern)
                         if path.is_file() and path.name.lower().endswith(suffix))
        if len(matches) != 1:
            raise ValueError("Expected one {} file for {}; found {}.".format(annotation_pattern, subject, len(matches)))
        subject_records = read_annotations(matches[0], subject, annotation_locale=annotation_locale)
        if not subject_records:
            raise ValueError("No gesture rows were found for {}.".format(subject))
        records.extend(subject_records)
    return records


def validate_annotations(records, require_reference_times=False):
    """Return errors/warnings; preserve every input record unchanged.

    Validation covers annotation structure and numbers only. Recording bounds,
    sensor clock alignment, and source-signal duplication need later IMU checks.
    """
    issues = []
    seen_intervals = {}
    seen_ids = set()
    for record in records:
        def add(code, message, severity="warning", timeline="", field=""):
            issues.append({
                "severity": severity, "code": code, "subject_id": record["subject_id"],
                "source_file": record["source_file"], "source_row": record["source_row"],
                "record_id": record["record_id"], "timeline": timeline,
                "field": field, "message": message,
            })
        if record["record_id"] in seen_ids:
            add("duplicate_record_id", "The same source row was loaded more than once.", "error")
        seen_ids.add(record["record_id"])
        for error in record["parse_errors"]:
            add(error["code"], "Cannot parse {!r}.".format(error["value"]), "error", field=error["field"])
        for field in ("subject_id", "session_id", "gesture_label", "object_id", "repetition"):
            if record[field] is None or record[field] == "":
                add("missing_identifier", "Required identifier is missing or invalid.", "error", field=field)
        for name, interval in record["intervals"].items():
            start, end = interval["start_s"], interval["end_s"]
            raw_fields = TIMELINES[name][:2]
            raw_missing = [not _clean(record["source_values"].get(field)) for field in raw_fields]
            if all(raw_missing):
                add("missing_interval", "Both time boundaries are missing; this timeline is unavailable.", timeline=name)
            elif any(raw_missing):
                add("incomplete_interval", "Only one time boundary is present.", "error", timeline=name)
            elif start is not None and end is not None and not interval["valid_interval"]:
                add("invalid_interval", "Require finite seconds with 0 <= start < end.", "error", timeline=name)
            if interval["valid_interval"]:
                reference = interval["reference_time"]
                if require_reference_times and reference is None:
                    add("missing_reference_time", "No reference clock is available in this session/step.", timeline=name)
                elif require_reference_times and not CLOCK_PATTERN.fullmatch(reference):
                    add("invalid_reference_format", "Reference clock format requires review; original text was preserved.", timeline=name)
                key = (record["subject_id"], record["session_id"], name, start, end)
                if key in seen_intervals:
                    add("duplicate_interval", "Same time bounds as {}; records were retained.".format(seen_intervals[key]), timeline=name)
                else:
                    seen_intervals[key] = record["record_id"]
    return issues


def summarize_annotations(records, issues):
    """Summarize subjects, gesture counts, usable intervals, and repetition groups."""
    subjects = {}
    groups = defaultdict(lambda: {"repetitions": set(), "segment_count": 0})
    for subject_id in sorted({record["subject_id"] for record in records}):
        subset = [record for record in records if record["subject_id"] == subject_id]
        subjects[subject_id] = {
            "segments": len(subset),
            "sessions": sorted({record["session_id"] for record in subset}),
            "gesture_counts": dict(sorted(Counter(record["gesture_label"] for record in subset).items())),
            "valid_intervals": {name: sum(record["intervals"][name]["valid_interval"] for record in subset) for name in TIMELINES},
            "segments_with_both_imu_intervals": sum(record["has_imu_pair"] for record in subset),
        }
    for record in records:
        key = (record["subject_id"], record["session_id"], record["gesture_label"], record["object_id"])
        groups[key]["segment_count"] += 1
        if record["repetition"] is not None:
            groups[key]["repetitions"].add(record["repetition"])
    return {
        "schema_version": 1,
        "annotation_locales": sorted({record["annotation_locale"] for record in records}),
        "subject_count": len(subjects),
        "segment_count": len(records),
        "gesture_labels": sorted({record["gesture_label"] for record in records}),
        "subjects": subjects,
        "validation": {
            "errors": sum(issue["severity"] == "error" for issue in issues),
            "warnings": sum(issue["severity"] == "warning" for issue in issues),
            "issue_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        },
        "repetition_groups": [
            dict(zip(("subject_id", "session_id", "gesture_label", "object_id"), key),
                 repetitions=sorted(value["repetitions"]), segment_count=value["segment_count"])
            for key, value in sorted(groups.items())
        ],
    }


def _within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def export_report(records, issues, summary, output_dir, dataset_root):
    """Write a NEW report directory under runs/imu/annotations/ only.

    Resolved paths also guard against directory links redirecting outputs into
    the source dataset. Existing directories/files are never overwritten.
    """
    output = _project_path(output_dir)
    allowed = (PROJECT_ROOT / "runs" / "imu" / "annotations").resolve()
    dataset = _project_path(dataset_root)
    if not _within(output, allowed) or _within(output, dataset) or _within(dataset, output):
        raise ValueError("Reports must be under runs/imu/annotations/ and outside the read-only dataset.")
    output.mkdir(parents=True, exist_ok=False)
    for filename, content in (("annotations.json", records), ("summary.json", summary)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    fields = ("severity", "code", "subject_id", "source_file", "source_row", "record_id", "timeline", "field", "message")
    with (output / "validation_issues.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)
    return output


def main(argv=None):
    """Read annotations, print a summary, and optionally export a guarded report."""
    parser = argparse.ArgumentParser(description="Read/validate annotations without changing the dataset.")
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--subjects", nargs="+", help="Subject folder names; default: all subjects.")
    parser.add_argument("--pattern", help="Annotation filename pattern; default: *_{locale}.csv using annotation_locale from config.")
    parser.add_argument("--export", action="store_true", help="Save a new report below runs/imu/annotations/.")
    args = parser.parse_args(argv)
    try:
        with _project_path(args.config).open("r", encoding="utf-8-sig") as stream:
            config = json.load(stream)
        if config.get("read_only") is not True:
            raise ValueError("Dataset configuration must keep read_only set to true.")
        dataset_root = _project_path(config.get("dataset_root", "../dataset"))
        records = read_dataset_annotations(
            dataset_root,
            subject_ids=args.subjects if args.subjects is not None else config.get("subject_ids"),
            annotation_pattern=args.pattern or config.get("annotation_file_pattern"),
            annotation_folder=config.get("folders", {}).get("annotations", "annotation"),
            annotation_locale=config.get("annotation_locale", "EN"),
        )
        issues = validate_annotations(records)
        summary = summarize_annotations(records, issues)
        print("Annotation locale: {}".format(", ".join(summary["annotation_locales"])))
        print("Subjects: {} | Gesture segments: {} | Labels: {}".format(
            summary["subject_count"], summary["segment_count"], ", ".join(summary["gesture_labels"])))
        for subject, counts in summary["subjects"].items():
            print("  {}: {} segments; both IMU intervals: {}; sEMG intervals: {}".format(
                subject, counts["segments"], counts["segments_with_both_imu_intervals"], counts["valid_intervals"]["semg"]))
        print("Validation: {} errors, {} warnings".format(summary["validation"]["errors"], summary["validation"]["warnings"]))
        for code, count in summary["validation"]["issue_counts"].items():
            print("  {}: {}".format(code, count))
        if args.export:
            run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output = export_report(records, issues, summary, "runs/imu/annotations/" + run_name, dataset_root)
            print("Report saved to {}".format(output.relative_to(PROJECT_ROOT)))
        return 2 if summary["validation"]["errors"] else 0
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
