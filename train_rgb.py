"""Prepare/train RGB gestures from the sibling dataset, with TSM or MVTN.

Run --help for camera, model and protocol choices. Paths are resolved relative
to this repository, not the terminal directory. All generated artifacts stay
under runs/rgb; original CSV annotations and recordings are read-only.
"""
from scripts.video_cli import main, train_video


def train(config_path="configs/rgb.json", protocol_path="configs/protocol2.json", **kwargs):
    """Programmatic equivalent of this RGB command-line entry point."""
    return train_video("rgb", config_path, protocol_path, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main("rgb"))
