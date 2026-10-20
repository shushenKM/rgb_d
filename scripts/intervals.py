"""Strict half-open selection on one sensor's own elapsed-time timeline.

The last sample owns a final nominal period: [0, 1) is covered by regular
100 Hz samples 0, .01, ..., .99. No annotation reference clocks are needed.
"""

import math

import numpy as np


def interval_sample_indices(timestamps_s, start_s, end_s, sample_rate_hz=None):
    """Select annotated samples in source order without a whole-file audit.

    Boolean selection avoids assuming that unselected timestamps are sorted.
    It neither changes timestamps nor sorts/removes source samples.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2 or not np.isfinite(timestamps).all():
        raise ValueError("A timestamp timeline must contain finite numeric samples.")
    for value in (start_s, end_s):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
            raise ValueError("Interval boundaries must be finite numeric seconds.")
    if not 0 <= start_s < end_s:
        raise ValueError("An interval requires 0 <= start < end.")
    steps = np.diff(timestamps)
    positive = steps[steps > 0]
    cadence = 1.0 / sample_rate_hz if sample_rate_hz is not None else float(np.median(positive)) if len(positive) else 0.0
    tolerance = max(1e-9, cadence * 1e-6)
    if start_s < timestamps.min() - tolerance or end_s > timestamps.max() + cadence + tolerance:
        raise ValueError("Annotation interval is outside recording sample coverage.")
    selected = np.flatnonzero((timestamps >= start_s - 1e-10) & (timestamps < end_s - 1e-10))
    if not len(selected):
        raise ValueError("Annotation interval contains no recorded samples.")
    return selected


def interval_indices(timestamps_s, start_s, end_s):
    """Return ``(left, right)`` bounds for ``start_s <= t < end_s``.

    Reject nonfinite/nonincreasing timelines, invalid boundaries, intervals
    outside recording coverage, and intervals with no actual samples. Bounds
    are never clipped, reordered, or copied from a different sensor.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("A timestamp timeline must contain at least two samples.")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing.")
    for value in (start_s, end_s):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
            raise ValueError("Interval boundaries must be finite numeric seconds.")
    if not 0 <= start_s < end_s:
        raise ValueError("An interval requires 0 <= start < end.")
    cadence = float(np.median(np.diff(timestamps)))
    tolerance = max(1e-9, cadence * 1e-6)
    if start_s < timestamps[0] - tolerance or end_s > timestamps[-1] + cadence + tolerance:
        raise ValueError("Annotation interval is outside recording sample coverage.")
    left = int(np.searchsorted(timestamps, start_s, side="left"))
    right = int(np.searchsorted(timestamps, end_s, side="left"))
    if left == right:
        raise ValueError("Annotation interval contains no recorded samples.")
    return left, right
