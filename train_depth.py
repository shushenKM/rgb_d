"""Prepare/train depth gestures separately from RGB using TSM or MVTN.

Read original uint16 PNG depth values and match them to annotated video-frame
indices through timestamps.csv. Depth scaling is configured in raw sensor units
in configs/depth.json. Source data is read-only; outputs stay under runs/depth.
"""
from scripts.video_cli import main, train_video


def train(config_path="configs/depth.json", protocol_path="configs/protocol2.json", **kwargs):
    """Programmatic equivalent of this depth command-line entry point."""
    return train_video("depth", config_path, protocol_path, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main("depth"))
