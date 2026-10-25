"""Prepare lazy RGB/depth gesture datasets using the IMU/sEMG split protocols.

PURPOSE
    Read the RGB-D CSV parents, validate selected camera recordings, assign
    entire subjects/repetition trials to partitions, then sample model clips.
    The returned manifests contain metadata and frame indices, not full video
    arrays. VideoGestureDataset decodes only a requested training/evaluation
    sample, so complete recordings are never loaded into model memory.

SAMPLING AND VIEWS
    Each ordinary gesture produces one fixed-length clip. Each long screw
    parent produces five nonoverlapping, evenly spaced approximately one-second
    clips. A short parent can explicitly shrink those five intervals, with a
    warning. All children and all selected views remain in the parent's split.
    Camera-specific annotated times are kept separate; no view's times are
    copied to another view. Output is [views, frames, channels, height, width].

PREPROCESSING
    RGB frames are resized to a configured square, scaled to [0,1], and use
    fixed ImageNet mean/std. Depth is read as uint16 and clipped/scaled using
    explicitly configured RAW depth bounds, not an assumed physical unit.
    Missing depth pixels (zero) remain zero. No statistics are fitted on test
    data. Frame-bin centers and spatial resizing are deterministic in this
    first baseline; training and evaluation share the same sampling rule.

DATA PROTECTION
    dataset_root resolves relative to this repository; all sources are opened
    only for reading. This module never writes files. The entrypoints save
    reports/checkpoints under runs/rgb or runs/depth. No source interval is
    repaired and missing data never falls back to another recording/frame.
"""

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from scripts.labels import build_label_mapping
from scripts.protocols import SplitValidationError, validate_split
from scripts.rgbd_annotations import VIEW_COLUMNS, read_dataset_rgbd_annotations
from scripts.training_protocols import build_training_splits
from vision.recordings import (inspect_recording, read_depth_frames, read_rgb_frames,
                               resolve_recording, sample_frame_indices)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "validation", "test")
SOURCE_FIELDS = ("record_id", "subject_id", "session_id", "object_id", "repetition",
                 "gesture_label", "source_file", "source_row")
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class VideoDatasetPreparationError(ValueError):
    """Preparation failure with reviewable issues and its partial manifest."""
    def __init__(self, issues, manifest=None):
        self.issues = issues
        self.manifest = [] if manifest is None else manifest
        codes = sorted({issue["code"] for issue in issues if issue.get("severity") == "error"})
        super().__init__("Video preparation requires review: {}".format(", ".join(codes)))


def _issue(issues, code, message, severity="error", record=None, **details):
    if record is not None:
        details.update({key: record.get(key) for key in SOURCE_FIELDS})
    issues.append(dict(details, code=code, message=message, severity=severity))


def _block(issues, manifest):
    if any(issue.get("severity") == "error" for issue in issues):
        raise VideoDatasetPreparationError(issues, manifest)


def video_settings(config):
    """Validate input settings shared by preparation and lazy decoding."""
    modality = config.get("modality", "rgb")
    if modality not in ("rgb", "depth"):
        raise ValueError("Video modality must be rgb or depth")
    views = config.get("views", ["front"])
    if (not isinstance(views, (list, tuple)) or not views or len(set(views)) != len(views)
            or any(view not in VIEW_COLUMNS for view in views)):
        raise ValueError("views must select unique camera names: front, above, left, right")
    inputs = config.get("input") or {}
    frames, size = inputs.get("num_frames", 8), inputs.get("image_size", 128)
    if type(frames) is not int or frames < 2:
        raise ValueError("input.num_frames must be an integer of at least two")
    if type(size) is not int or size < 32:
        raise ValueError("input.image_size must be an integer of at least 32")
    low, high = inputs.get("depth_min", 0), inputs.get("depth_max", 5000)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high)) or not 0 <= low < high:
        raise ValueError("Depth clipping requires finite 0 <= input.depth_min < input.depth_max")
    return dict(modality=modality, views=list(views), num_frames=frames, image_size=size,
                depth_min=float(low), depth_max=float(high))


def _screw_settings(config):
    settings = config.get("screw_sampling") or {}
    duration = settings.get("duration_s", 1.0)
    count = settings.get("segments_per_trial", 5)
    policy = settings.get("short_parent_policy", "shrink")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("screw_sampling.duration_s must be finite and positive")
    if type(count) is not int or count != 5:
        raise ValueError("screw_sampling.segments_per_trial must be five for 30 screw clips per subject")
    if policy not in ("shrink", "error"):
        raise ValueError("screw_sampling.short_parent_policy must be shrink or error")
    return float(duration), count, policy


def _select_records(records, config, issues):
    sessions = config.get("session_ids", ["step1", "step2", "step3"])
    if not isinstance(sessions, list) or not sessions or len(set(sessions)) != len(sessions):
        raise ValueError("session_ids must select unique annotation session names")
    views = video_settings(config)["views"]
    selected = []
    for original in records:
        if original["session_id"] not in sessions:
            continue
        record = deepcopy(original)
        for field in ("record_id", "subject_id", "session_id", "object_id", "gesture_label"):
            if not isinstance(record.get(field), str) or not record[field]:
                _issue(issues, "missing_annotation_identifier", "Missing {}".format(field), record=record)
        if type(record.get("repetition")) is not int or record["repetition"] < 1:
            _issue(issues, "invalid_repetition", "Repetition must be a positive integer", record=record)
        selected_columns = {c for view in views for c in VIEW_COLUMNS[view][:2]}
        for error in record.get("parse_errors", []):
            if "field" not in error or error["field"] in selected_columns | {"repetition"}:
                _issue(issues, error["code"], "Cannot parse selected annotation values", record=record, detail=error)
        record["intervals"] = {view: record["intervals"][view] for view in views}
        for view, interval in record["intervals"].items():
            start, end = interval["start_s"], interval["end_s"]
            if start is None or end is None or not 0 <= start < end:
                _issue(issues, "invalid_video_interval", "Selected view requires finite 0 <= start < end", record=record, view=view)
        selected.append(record)
    if not selected:
        _issue(issues, "no_selected_annotations", "No gesture records remain in the selected sessions")
    for session in sessions:
        if not any(r["session_id"] == session for r in selected):
            _issue(issues, "missing_selected_session", "No annotations for selected session", session_id=session)
    return selected


def _parent_clips(record, recordings, settings, screw_settings, label_mapping, issues):
    requested, count, short_policy = screw_settings
    count = count if record["gesture_label"] == "screw" else 1
    intervals = {}
    for view in settings["views"]:
        parent = record["intervals"][view]
        start, end = parent["start_s"], parent["end_s"]
        if count == 1:
            intervals[view] = [(start, end, end-start)]
            continue
        duration = requested
        if end-start < count * duration - 1e-9:
            if short_policy == "error":
                raise ValueError("Screw parent is too short for five nonoverlapping {}-second clips ({})".format(duration, view))
            duration = (end-start)/count
            _issue(issues, "short_screw_parent", "Five screw clips use a shorter duration inside this parent", "warning", record,
                   view=view, requested_duration_s=requested, effective_duration_s=duration)
        starts = np.linspace(start, end-duration, count)
        intervals[view] = [(float(s), min(float(s+duration), end), duration) for s in starts]
    clips = []
    for child in range(count):
        sample = {key: record[key] for key in SOURCE_FIELDS}
        sample.update(sample_id="{}#clip{:02d}".format(record["record_id"], child),
                      parent_record_id=record["record_id"], label=label_mapping[record["gesture_label"]],
                      clip_index=child, clips_in_parent=count, views=[])
        for view in settings["views"]:
            recording = recordings[(record["subject_id"], record["session_id"], view)]
            start, end, duration = intervals[view][child]
            indices = sample_frame_indices(start, end, recording["fps"], recording["frame_count"], settings["num_frames"])
            data = dict(view=view, video_path=recording["video_path"], start_s=start, end_s=end,
                        parent_start_s=record["intervals"][view]["start_s"],
                        parent_end_s=record["intervals"][view]["end_s"],
                        effective_duration_s=duration, frame_indices=indices, fps=recording["fps"])
            if settings["modality"] == "depth":
                paths = [recording["depth_map"].get(index) for index in indices]
                if any(path is None for path in paths):
                    missing = [index for index, path in zip(indices, paths) if path is None]
                    raise ValueError("Missing depth image for selected RGB frame indices {} in {}".format(missing, recording["video_path"]))
                data["depth_paths"] = paths
            sample["views"].append(data)
        clips.append(sample)
    return clips


def _training_splits(records, config, protocol):
    """Use the shared grouping policy, with one explicit temporal exception.

    The opt-in only downgrades SAME-subject source-time intersections involving
    the test partition. Duplicate IDs and trials crossing partitions remain
    errors. It never changes assignments, annotations, clips, or sample values.
    Cross-subject signal copies are not content-hashed by this video baseline.
    """
    training = config.get("training") or {}
    quality = config.get("data_quality") or {}
    allow = quality.get("allow_within_subject_test_overlap", False)
    if type(allow) is not bool:
        raise ValueError("data_quality.allow_within_subject_test_overlap must be boolean")
    if not allow:
        return build_training_splits(records, protocol, training.get("validation"), training.get("seed", 42))
    # Trial selection depends on identifiers/labels, not interval values. Defer
    # only the temporal check until the shared grouping logic has completed.
    grouped = [dict(record, intervals={}) for record in records]
    folds = build_training_splits(grouped, protocol, training.get("validation"), training.get("seed", 42))
    originals = {r["record_id"]: r for r in records}
    for fold in folds:
        for part in (*PARTITIONS, "excluded"):
            fold[part] = [originals[r["record_id"]] for r in fold[part]]
        validation = fold["metadata"]["validation"]
        checked = validate_split(fold, allow_train_validation_overlap=validation.get("allow_train_validation_overlap", False))
        for issue in checked:
            if issue["code"] == "overlapping_intervals":
                if "test" in issue.get("partitions", []):
                    issue = dict(issue, severity="warning", allowed_by_configuration=True,
                                 accepted_policy="data_quality.allow_within_subject_test_overlap")
                fold["issues"].append(issue)
        fold["metadata"]["allow_within_subject_test_overlap"] = True
        fold["valid"] = not any(issue["severity"] == "error" for issue in fold["issues"])
        if not fold["valid"]:
            raise SplitValidationError(fold)
    return folds


def prepare_video_dataset(dataset_config, config, protocol):
    """Return validated lazy clip manifests and folds, without writing artifacts."""
    if dataset_config.get("read_only") is not True:
        raise ValueError("dataset_config.read_only must remain true")
    root = (PROJECT_ROOT / dataset_config.get("dataset_root", "../dataset")).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(root))
    settings = video_settings(config)
    screw = _screw_settings(config)
    issues, manifest = [], []
    subjects = config.get("subject_ids")
    try:
        records = read_dataset_rgbd_annotations(root, dataset_config, subjects)
    except (OSError, ValueError) as error:
        _issue(issues, "annotation_read_error", str(error))
        _block(issues, manifest)
    records = _select_records(records, config, issues)
    _block(issues, manifest)
    labels = build_label_mapping(r["gesture_label"] for r in records)
    expected_labels = config.get("expected_gesture_labels")
    if expected_labels is not None and set(labels) != set(expected_labels):
        _issue(issues, "unexpected_gesture_vocabulary", "Gesture vocabulary differs from configuration", observed=sorted(labels))
    expected_count = config.get("expected_class_count", 9)
    if expected_count is not None and len(labels) != expected_count:
        _issue(issues, "unexpected_class_count", "Expected {} classes, found {}".format(expected_count, len(labels)))
    _block(issues, manifest)
    try:
        # This is intentionally before clips, frames, model data, or view expansion.
        parent_folds = _training_splits(records, config, protocol)
    except SplitValidationError as error:
        issues.extend(error.result["issues"])
        _block(issues, manifest)
    except ValueError as error:
        _issue(issues, "invalid_training_split", str(error))
        _block(issues, manifest)
    for fold in parent_folds:
        issues.extend(dict(issue, fold=fold["name"]) for issue in fold.get("issues", []))
    recordings = {}
    for subject, session, view in sorted({(r["subject_id"], r["session_id"], view) for r in records for view in settings["views"]}):
        try:
            folder = resolve_recording(root, dataset_config, subject, session, view)
            item = inspect_recording(folder, root, settings["modality"], probe_frames=True)
            recordings[(subject, session, view)] = item
            manifest.append(dict({k: v for k, v in item.items() if k != "depth_map"},
                                 subject_id=subject, session_id=session, view=view))
            if item.get("missing_depth_frame_count", 0):
                _issue(issues, "missing_depth_outside_or_inside_annotations", "Recording contains absent depth frames; selected clip frames are checked separately", "warning",
                       subject_id=subject, session_id=session, view=view, count=item["missing_depth_frame_count"])
        except (OSError, ValueError) as error:
            _issue(issues, "recording_unavailable", str(error), subject_id=subject, session_id=session, view=view)
    _block(issues, manifest)
    samples = {}
    probed_depth = set()
    # Build each assigned parent once; excluded protocol repetitions stay excluded.
    assigned = {r["record_id"]: r for f in parent_folds for part in PARTITIONS for r in f[part]}
    for record in assigned.values():
        try:
            samples[record["record_id"]] = _parent_clips(record, recordings, settings, screw, labels, issues)
            if settings["modality"] == "depth":
                for view in samples[record["record_id"]][0]["views"]:
                    if view["video_path"] not in probed_depth:
                        read_depth_frames(root, [view["depth_paths"][0]], settings["image_size"])
                        probed_depth.add(view["video_path"])
        except (OSError, ValueError) as error:
            _issue(issues, "invalid_clip_source", str(error), record=record)
    _block(issues, manifest)
    normalization = (dict(method="fixed_imagenet", mean=list(RGB_MEAN), std=list(RGB_STD), fitted=False)
                     if settings["modality"] == "rgb" else
                     dict(method="fixed_depth_range", minimum=settings["depth_min"], maximum=settings["depth_max"],
                          units="raw_recorded_depth_values", zero_is_missing=True, fitted=False))
    folds = []
    for original in parent_folds:
        splits = {part: [sample for parent in original[part] for sample in samples[parent["record_id"]]] for part in PARTITIONS}
        summary = {part: dict(parents=len(original[part]), clips=len(items),
                              subjects=sorted({s["subject_id"] for s in items}),
                              gesture_counts=dict(sorted(Counter(s["gesture_label"] for s in items).items())))
                   for part, items in splits.items()}
        folds.append(dict(name=original["name"], splits=splits, split_summary=summary,
                          protocol_metadata=original["metadata"], normalization=normalization,
                          excluded_records=[r["record_id"] for r in original["excluded"]]))
    return dict(valid=True, issues=issues, manifest=manifest, label_mapping=labels, folds=folds,
                dataset_root=root, input_settings=settings, annotation_count=len(records))


class VideoGestureDataset:
    """Lazy map-style dataset returning (float tensor[V,T,C,H,W], class_index).

    Torch is imported only when retrieving tensors; metadata-only preparation
    does not require a CUDA build or model dependencies. The training argument
    is retained for future augmentation; this baseline samples deterministically.
    """
    def __init__(self, samples, dataset_root, config, training=False):
        self.samples = list(samples)
        self.dataset_root = Path(dataset_root).resolve()
        self.settings = video_settings(config)
        self.training = bool(training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        sample = self.samples[index]
        settings = self.settings
        if [v["view"] for v in sample["views"]] != settings["views"]:
            raise ValueError("Sample view order differs from configured view order")
        arrays = []
        for view in sample["views"]:
            if len(view["frame_indices"]) != settings["num_frames"]:
                raise ValueError("Sample temporal length differs from configuration")
            if settings["modality"] == "rgb":
                array = read_rgb_frames(self.dataset_root, view["video_path"], view["frame_indices"], settings["image_size"])
                array = array.astype(np.float32) / 255.0
                array = (array - np.asarray(RGB_MEAN, dtype=np.float32)) / np.asarray(RGB_STD, dtype=np.float32)
                array = array.transpose(0, 3, 1, 2)
            else:
                paths = view.get("depth_paths")
                if paths is None or len(paths) != settings["num_frames"]:
                    raise ValueError("Sample has no matching selected depth paths")
                raw = read_depth_frames(self.dataset_root, paths, settings["image_size"])
                array = (np.clip(raw.astype(np.float32), settings["depth_min"], settings["depth_max"])
                         - settings["depth_min"]) / (settings["depth_max"] - settings["depth_min"])
                array[raw == 0] = 0.0
                array = array[:, None, :, :]
            arrays.append(array)
        result = np.stack(arrays).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Decoded model input is nonfinite")
        return torch.from_numpy(np.ascontiguousarray(result)), int(sample["label"])
