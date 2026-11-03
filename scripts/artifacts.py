"""Portable config loading and modality outputs confined to this project's runs/."""
import json
import hashlib
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


def experiment_run_name(modality, config, protocol, *, prepare_only=False, synthetic=False):
    """Describe effective settings after CLI overrides; never use the clock.

    Epochs is the requested limit, even if early stopping finishes sooner.
    Full settings remain in config.json. Keep names short enough for Windows;
    unusually long names receive a content hash after truncation.
    """
    video = modality in ("rgb", "depth")
    training, inputs = config.get("training") or {}, config.get("input") or {}
    defaults = {"imu": "If_ConvTransformer_W", "semg": "VisionTransformer",
                "rgb": "tsm", "depth": "tsm"}
    def token(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-_") or "unknown"
    parts = [modality, config.get("model", defaults[modality]),
             "batch{}".format(training.get("batch_size", 4 if video else 64)),
             "epoch{}".format(training.get("epochs", 30)),
             protocol.get("name", protocol.get("type", "protocol"))]
    key = "views" if video else "sensor_ids" if modality == "imu" else "grid_ids"
    selection = config.get(key) or []
    if selection:
        parts.append("-".join(token(item) for item in selection))
    if video:
        parts.extend(["frames{}".format(inputs.get("num_frames", 8)),
                      "size{}".format(inputs.get("image_size", 128))])
    else:
        parts.append("points{}".format(inputs.get("num_points", 128)))
    parts.extend(["lr{}".format(training.get("learning_rate", .0001 if video else .0005)),
                  "seed{}".format(training.get("seed", 42))])
    for key, prefix in (("test_subject_id", "test"), ("validation_subject_id", "val")):
        if protocol.get(key):
            parts.append(prefix + "-" + token(protocol[key]))
    if synthetic:
        parts.append("synthetic")
    if prepare_only:
        parts.append("prepare")
    name = "_".join(token(part) for part in parts)
    if len(name) > 160:
        name = name[:147].rstrip("._-") + "_" + hashlib.sha256(name.encode()).hexdigest()[:12]
    return name


def create_run_directory(output_root="runs/imu", name=None, project_root=None, *, modality="imu", default_name=None):
    """Create a new run; auto names get numeric suffixes, explicit names must be unused."""
    if modality not in ("imu", "semg", "rgb", "depth"):
        raise ValueError("Output modality must be imu, semg, rgb or depth")
    root = (PROJECT_ROOT if project_root is None else Path(project_root)).resolve()
    output = project_path(output_root, root)
    allowed = (root / "runs" / modality).resolve()
    try:
        allowed.relative_to(root)
        output.relative_to(allowed)
    except ValueError:
        raise ValueError("{} outputs must stay under this project's runs/{} directory".format(modality.upper(), modality))
    automatic = name is None
    name = (default_name or modality + "_run") if automatic else name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run name must be one directory name containing letters, numbers, '.', '_' or '-'")
    output.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = name if number == 1 else "{}_run{:02d}".format(name, number)
        run = output / candidate
        try:
            run.mkdir(parents=True, exist_ok=False)
            return run
        except FileExistsError:
            if not automatic:
                raise FileExistsError("Run already exists; choose a new name: {}".format(name))
            # Atomic mkdir also protects concurrent identical experiments.
            number += 1


def fold_directory(run, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid fold name: {!r}".format(name))
    path = Path(run) / name
    path.mkdir(exist_ok=False)
    return path
