"""Read RGB video metadata and RGB-to-depth frame maps without source writes.

Annotations use the AVI playback clock. A zero-based decoded frame index is
matched to the one-based color_idx in timestamps.csv, then depth_saved_idx
selects its PNG. Device timestamps are NOT treated as video playback seconds.
All paths recorded in manifests are relative to the portable dataset root.
"""

import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np


def source_path(root, relative):
    """Resolve a source only inside dataset_root, including symlink checks."""
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("A recording path escapes dataset_root: {}".format(relative))
    return result


def resolve_recording(dataset_root, dataset_config, subject, session, view):
    mapping = dataset_config.get("session_mapping") or {"step1": "session1", "step2": "session2", "step3": "session3"}
    if session not in mapping:
        raise ValueError("No recording session mapping for {}".format(session))
    folder = dataset_config.get("folders", {}).get("rgbd", "rgbd")
    return source_path(dataset_root, Path(subject) / folder / mapping[session] / view)


def inspect_recording(folder, dataset_root, modality="rgb", probe_frames=True):
    """Check container metadata and optional endpoint decoding; depth adds its map.

    This verifies selected source availability, not every pixel in a recording.
    No raw arrays are cached or written. Missing/undecodable files raise errors.
    """
    root = Path(dataset_root).resolve()
    folder = source_path(root, folder)
    video = source_path(root, folder / "color.avi")
    if not video.is_file():
        raise FileNotFoundError("RGB recording is missing: {}".format(video))
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open RGB recording: {}".format(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid video metadata: {}".format(video))
        if probe_frames:
            for frame in sorted(set((0, count-1))):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                success, image = capture.read()
                if not success or image is None:
                    raise ValueError("Cannot decode frame {} of {}".format(frame, video))
    finally:
        capture.release()
    stat = video.stat()
    result = {"video_path": video.relative_to(root).as_posix(), "fps": fps,
              "frame_count": count, "duration_s": count / fps, "width": width, "height": height,
              "video_size_bytes": stat.st_size, "video_mtime_ns": stat.st_mtime_ns,
              "clock": "video_playback", "frame_index_origin": 0}
    if modality == "depth":
        timestamp_file = source_path(root, folder / "timestamps.csv")
        depth_folder = source_path(root, folder / "depth_raw")
        if not depth_folder.is_dir():
            raise FileNotFoundError("Depth directory is missing: {}".format(depth_folder))
        names = {}
        relative_folder = depth_folder.relative_to(root).as_posix()
        # DirEntry caches Windows file-type information, avoiding a separate
        # stat call for each of the hundreds of thousands of depth images.
        with os.scandir(depth_folder) as entries:
            for entry in entries:
                stem, suffix = os.path.splitext(entry.name)
                if suffix.lower() == ".png" and stem.isdigit() and entry.is_file():
                    index = int(stem)
                    if index in names:
                        raise ValueError("Ambiguous depth filename index {} in {}".format(index, depth_folder))
                    names[index] = relative_folder + "/" + entry.name
        mapping = {}
        missing = []
        with timestamp_file.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"color_idx", "depth_saved_idx", "depth_saved"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Depth mapping columns are missing in {}".format(timestamp_file))
            for line, row in enumerate(reader, 2):
                try:
                    color = int(row["color_idx"])
                    saved = int(row["depth_saved"])
                    if color < 1 or color > count or color-1 in mapping or saved not in (0, 1):
                        raise ValueError()
                    if saved:
                        depth = int(row["depth_saved_idx"])
                        if depth <= 0:
                            raise ValueError()
                        mapping[color-1] = names.get(depth)
                        if mapping[color-1] is None:
                            missing.append(color-1)
                    else:
                        mapping[color-1] = None
                        missing.append(color-1)
                except (ValueError, KeyError):
                    raise ValueError("Invalid RGB/depth mapping at {}:{}".format(timestamp_file, line)) from None
        if set(mapping) != set(range(count)):
            raise ValueError("Timestamp color_idx does not cover every RGB frame in {}".format(timestamp_file))
        result.update(timestamp_path=timestamp_file.relative_to(root).as_posix(),
                      depth_map=mapping, depth_image_count=len(names), missing_depth_frame_count=len(missing),
                      missing_depth_frame_examples=missing[:10])
    return result


def sample_frame_indices(start_s, end_s, fps, frame_count, num_frames):
    """Sample bin centers from frames whose playback timestamps lie in [start,end).

    Short clips repeat selected indices to keep a fixed input length. This does
    not create new source trials or alter the train/validation/test partition.
    """
    if (not all(math.isfinite(v) for v in (start_s, end_s, fps))
            or not 0 <= start_s < end_s or fps <= 0):
        raise ValueError("Invalid video interval or frame rate")
    if type(frame_count) is not int or frame_count < 1 or type(num_frames) is not int or num_frames < 1:
        raise ValueError("Frame counts must be positive integers")
    if end_s > frame_count / fps + 1e-7:
        raise ValueError("Annotation interval [{}, {}) exceeds video duration {}".format(start_s, end_s, frame_count/fps))
    left = int(math.ceil(start_s * fps - 1e-8))
    right = min(frame_count, int(math.ceil(end_s * fps - 1e-8)))
    if left < 0 or right <= left:
        raise ValueError("Annotation interval contains no RGB frame timestamps")
    positions = left + np.floor((np.arange(num_frames) + .5) * (right-left) / num_frames).astype(np.int64)
    return positions.tolist()


def read_rgb_frames(dataset_root, video_path, indices, image_size):
    """Decode requested RGB indices, raising on failure instead of substituting."""
    path = source_path(dataset_root, video_path)
    capture = cv2.VideoCapture(str(path))
    frames = {}
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open RGB video: {}".format(path))
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("Cannot decode frame {} of {}".format(index, path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames[index] = frame
    finally:
        capture.release()
    return np.stack([frames[index] for index in indices], axis=0)


def read_depth_frames(dataset_root, depth_paths, image_size):
    """Preserve native uint16 depth until caller-specified clipping/scaling."""
    frames = {}
    for relative in dict.fromkeys(depth_paths):
        path = source_path(dataset_root, relative)
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError("Cannot decode depth image: {}".format(path))
        if frame.ndim != 2 or frame.dtype != np.uint16:
            raise ValueError("Depth must be a single-channel uint16 image: {}".format(path))
        frames[relative] = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return np.stack([frames[path] for path in depth_paths], axis=0)
