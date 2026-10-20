"""Portable config loading and modality outputs confined to this project's runs/."""
import json
from datetime import datetime, timezone
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value, project_root=None):
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def read_config(value, project_root=None):
    with project_path(value, project_root).open("r", encoding="utf-8-sig") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    return config


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot serialize {}".format(type(value).__name__))


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
        stream.write("\n")


def create_run_directory(output_root="runs/imu", name=None, project_root=None, *, modality="imu"):
    if modality not in ("imu", "semg"):
        raise ValueError("Output modality must be imu or semg")
    root = (PROJECT_ROOT if project_root is None else Path(project_root)).resolve()
    output = project_path(output_root, root)
    allowed = (root / "runs" / modality).resolve()
    try:
        allowed.relative_to(root)
        output.relative_to(allowed)
    except ValueError:
        raise ValueError("{} outputs must stay under this project's runs/{} directory".format(modality.upper(), modality))
    name = name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run name must be one directory name containing letters, numbers, '.', '_' or '-'")
    run = output / name
    if run.exists():
        raise FileExistsError("Run already exists; choose a new name: {}".format(name))
    run.mkdir(parents=True, exist_ok=False)
    return run


def fold_directory(run, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid fold name: {!r}".format(name))
    path = Path(run) / name
    path.mkdir(exist_ok=False)
    return path
