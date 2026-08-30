"""Swappable end-effector bodies for Kitov deploy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EE_BODY_ROOT = REPO_ROOT / "ee_body"
EE_BODY_CONFIG_ROOT = EE_BODY_ROOT / "config"


def resolve_ee_body_config(ref: str | Path | None) -> Path | None:
    if ref is None:
        return None
    ref_text = str(ref).strip()
    if ref_text.lower() in {"", "none", "null", "false"}:
        return None

    path = Path(ref_text).expanduser()
    if path.is_absolute():
        return path
    if path.suffix == ".json" or "/" in ref_text:
        return (REPO_ROOT / path).resolve(strict=False)
    return EE_BODY_CONFIG_ROOT / f"{ref_text}.json"


def load_ee_body(ref: str | Path | None) -> Any | None:
    config_path = resolve_ee_body_config(ref)
    if config_path is None:
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    ee_type = str(payload.get("type", ""))
    if ee_type == "openarm_can_gripper":
        from ee_body.drivers.openarm_can_gripper import OpenArmCANGripperEE

        return OpenArmCANGripperEE.from_config(config_path, payload)
    raise ValueError(f"Unsupported ee_body type {ee_type!r} in {config_path}")
