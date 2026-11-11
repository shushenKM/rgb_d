"""Resolve portable protocol names and select protocol 2's active split mode.

Users edit one protocol2.json: grouped_percentages=false uses repetition IDs;
true uses split_percentages. Inactive settings remain in the saved configuration
but do not control assignment. Old percentage-only configs and the short alias
protocol2_percentages are supported for compatibility. No dataset access occurs.
"""
from copy import deepcopy
import math


def resolve_protocol_reference(value, folder="configs"):
    """Return (JSON path, percentage-alias flag), retaining explicit JSON paths."""
    value = str(value)
    normalized = value.replace("\\", "/")
    old_alias = normalized in (
        "protocol2_percentages", "protocol2_percentages.json",
        folder.rstrip("/") + "/protocol2_percentages.json")
    if old_alias:
        return folder.rstrip("/") + "/protocol2.json", True
    if "/" not in normalized and not value.endswith(".json"):
        return folder.rstrip("/") + "/" + value + ".json", False
    return value, False


def normalize_protocol(protocol, *, force_percentages=False):
    """Copy settings and derive the effective type without changing source JSON."""
    if not isinstance(protocol, dict):
        raise ValueError("Protocol configuration must be a JSON object")
    config = deepcopy(protocol)
    flag = config.get("grouped_percentages")
    if "grouped_percentages" in config and type(flag) is not bool:
        raise ValueError("grouped_percentages must be true or false (a JSON boolean)")
    kind = config.get("type")
    protocol2_types = ("held_out_repetitions", "held_out_macro_repetitions", "grouped_percentages")
    if kind not in protocol2_types:
        if flag is True or force_percentages:
            raise ValueError("grouped_percentages applies only to protocol2")
        return config
    if force_percentages:
        config["grouped_percentages"] = flag = True
    if flag is not None:
        config["type"] = "grouped_percentages" if flag else (
            "held_out_repetitions" if kind == "grouped_percentages" else kind)
    return config


def percentage_settings(protocol, seed=42):
    """Validate the active ratios; train/test positive, validation may be zero."""
    parts = ("train", "validation", "test")
    percentages = protocol.get("split_percentages")
    if not isinstance(percentages, dict) or set(percentages) != set(parts):
        raise ValueError("split_percentages must contain exactly train, validation and test")
    for value in percentages.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("split_percentages values must be finite numbers between 0 and 100")
    if percentages["train"] <= 0 or percentages["test"] <= 0:
        raise ValueError("split_percentages train and test must be positive")
    if not math.isclose(sum(percentages.values()), 100.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("split_percentages must sum to 100")
    split_seed = protocol.get("split_seed", seed)
    if type(split_seed) is not int or not 0 <= split_seed < 2**63:
        raise ValueError("split_seed must be a nonnegative integer smaller than 2**63")
    return dict(percentages), split_seed
