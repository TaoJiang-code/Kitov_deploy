#!/usr/bin/env python3
"""Replay a saved XRobot/PICO frame through GMR retargeting."""

from __future__ import annotations
'''
python scripts/debug/replay_xrobot_frame.py date/20260725_164701_673558_g1_frame003344.json \
  --robot bumi \
  --offset-to-ground \
  --viewer \
  --show-human \
  --quiet-gmr
'''
import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT, OnlineGMRRetargeter, ROBOT_CONFIGS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget a saved XRobot/PICO frame to G1 or BUMI.")
    parser.add_argument("frame_json", type=Path, help="Saved JSON produced by scripts/debug/xrobot_retarget.py P-key capture.")
    parser.add_argument("--robot", choices=sorted(ROBOT_CONFIGS), default="bumi", help="Target robot alias.")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--hz", type=float, default=50.0, help="Viewer refresh rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer duration. 0 means run until Ctrl-C.")
    parser.add_argument("--actual-human-height", type=float, default=None, help="Optional human height used to scale IK targets.")
    parser.add_argument("--solver", default="daqp", help="IK solver passed to mink/qpsolvers.")
    parser.add_argument("--damping", type=float, default=5e-1, help="IK damping passed to GMR.")
    parser.add_argument("--ground-offset", type=float, default=0.0, help="Subtract this z offset from all human targets.")
    parser.add_argument("--offset-to-ground", action="store_true", help="Shift the saved frame so the lowest foot target sits above ground.")
    parser.add_argument("--use-velocity-limit", action="store_true", help="Enable GMR velocity limits.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Repeated IK steps on the same frame before display.")
    parser.add_argument("--viewer", action="store_true", help="Open GMR MuJoCo viewer.")
    parser.add_argument("--show-human", action="store_true", help="Draw GMR human targets in the viewer.")
    parser.add_argument("--show-all-human", action="store_true", help="Draw all converted XRobot body joints, including joints not used by IK.")
    parser.add_argument("--human-axes-only", action="store_true", help="Draw human target axes without blue spheres.")
    parser.add_argument("--show-human-name", action="store_true", help="Draw human joint names next to the target axes.")
    parser.add_argument("--quiet-gmr", action="store_true", help="Suppress GMR model/body/dof listing.")
    parser.add_argument("--print-qpos", choices=["none", "root", "all"], default="root", help="How much qpos to print.")
    return parser.parse_args()


def _load_converted_body(path: Path) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    converted = payload.get("converted_body")
    if not isinstance(converted, dict):
        raise RuntimeError(f"Frame JSON does not contain converted_body: {path}")

    body: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, pose in converted.items():
        try:
            pos = np.asarray(pose["position_rhs"], dtype=np.float64)
            quat = np.asarray(pose["quat_wxyz_rhs"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid converted_body entry for joint {name!r}") from exc
        if pos.shape != (3,) or quat.shape != (4,):
            raise RuntimeError(f"Invalid converted_body shape for joint {name!r}: pos={pos.shape}, quat={quat.shape}")
        body[name] = (pos, quat)

    return payload, body


def _format_qpos(qpos: np.ndarray, mode: str, *, has_floating_base: bool) -> str:
    if mode == "none":
        return ""
    if mode == "root" and has_floating_base:
        root_pos = ", ".join(f"{x:.3f}" for x in qpos[:3])
        root_quat = ", ".join(f"{x:.4f}" for x in qpos[3:7])
        return f"root_pos=[{root_pos}] root_quat_wxyz=[{root_quat}]"
    if mode == "root":
        values = ", ".join(f"{x:.4f}" for x in qpos[: min(8, len(qpos))])
        return f"qpos_head=[{values}]"
    values = ", ".join(f"{x:.5f}" for x in qpos)
    return f"qpos=[{values}]"


def main() -> int:
    args = _parse_args()
    payload, body = _load_converted_body(args.frame_json)

    retargeter = OnlineGMRRetargeter(
        args.robot,
        gmr_root=args.gmr_root,
        actual_human_height=args.actual_human_height,
        solver=args.solver,
        damping=args.damping,
        use_velocity_limit=args.use_velocity_limit,
        verbose=not args.quiet_gmr,
    )
    retargeter.set_ground_offset(args.ground_offset)

    qpos = None
    for _ in range(max(args.warmup_steps, 1)):
        qpos = retargeter.retarget(body, offset_to_ground=args.offset_to_ground)
    assert qpos is not None

    print(
        "[replay_xrobot_frame] "
        f"input={args.frame_json} saved_robot={payload.get('robot')} "
        f"target_robot={args.robot} joints={len(body)} qpos={len(qpos)} "
        f"{_format_qpos(qpos, args.print_qpos, has_floating_base=retargeter.has_floating_base)}"
    )

    if not args.viewer:
        return 0

    stop = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    viewer = retargeter.make_viewer(motion_fps=args.hz)
    start_time = time.monotonic()
    try:
        while not stop:
            if args.duration > 0.0 and time.monotonic() - start_time >= args.duration:
                break
            if args.show_human:
                human_motion = retargeter.prepare_debug_human_data(
                    body,
                    offset_to_ground=args.offset_to_ground,
                    include_unscaled=args.show_all_human,
                )
            else:
                human_motion = None
            viewer.step_qpos(
                qpos,
                human_motion_data=human_motion,
                show_human_body_name=args.show_human_name,
                show_human_points=not args.human_axes_only,
                rate_limit=True,
            )
    finally:
        viewer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
