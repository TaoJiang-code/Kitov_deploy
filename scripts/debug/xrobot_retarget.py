#!/usr/bin/env python3
"""Online XRobot/PICO to robot qpos retargeting."""

from __future__ import annotations

'''
python scripts/debug/xrobot_retarget.py --robot g1 --viewer --hz 50 --quiet-gmr --save-dir date

python scripts/debug/xrobot_retarget.py --robot bumi --viewer --hz 50 --quiet-gmr --save-dir date
'''

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT, OnlineGMRRetargeter, ROBOT_CONFIGS
from kitov_deploy.xrobot_stream import XRobotBodyFrame, XRobotBodyStreamer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget live XRobot/PICO body data to robot qpos.")
    parser.add_argument("--robot", choices=sorted(ROBOT_CONFIGS), default="bumi", help="Target robot alias.")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--hz", type=float, default=50.0, help="Polling and viewer rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds. 0 means run until Ctrl-C.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument("--actual-human-height", type=float, default=None, help="Optional human height used to scale IK targets.")
    parser.add_argument("--solver", default="daqp", help="IK solver passed to mink/qpsolvers.")
    parser.add_argument("--damping", type=float, default=5e-1, help="IK damping passed to GMR.")
    parser.add_argument("--ground-offset", type=float, default=0.0, help="Subtract this z offset from all human targets.")
    parser.add_argument("--offset-to-ground", action="store_true", help="Shift each frame so the lowest foot target sits above ground.")
    parser.add_argument("--use-velocity-limit", action="store_true", help="Enable GMR velocity limits.")
    parser.add_argument("--viewer", action="store_true", help="Open GMR MuJoCo viewer.")
    parser.add_argument("--show-human", action="store_true", help="Draw GMR human targets in the viewer.")
    parser.add_argument("--show-all-human", action="store_true", help="Draw all converted XRobot body joints, including joints not used by IK.")
    parser.add_argument("--human-axes-only", action="store_true", help="Draw human target axes without blue spheres.")
    parser.add_argument("--show-human-name", action="store_true", help="Draw human joint names next to the target axes.")
    parser.add_argument("--quiet-gmr", action="store_true", help="Suppress GMR model/body/dof listing.")
    parser.add_argument("--print-qpos", choices=["none", "root", "all"], default="root", help="How much qpos to print.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "recordings" / "xrobot_frames",
        help="Directory for P-key XRobot frame saves in viewer mode.",
    )
    return parser.parse_args()


def _format_qpos(qpos: np.ndarray, mode: str, *, has_floating_base: bool) -> str:
    if mode == "none":
        return ""
    if mode == "root" and has_floating_base:
        root_pos = ", ".join(f"{x:.3f}" for x in qpos[:3])
        root_quat = ", ".join(f"{x:.4f}" for x in qpos[3:7])
        return f" root_pos=[{root_pos}] root_quat_wxyz=[{root_quat}]"
    if mode == "root":
        values = ", ".join(f"{x:.4f}" for x in qpos[: min(8, len(qpos))])
        return f" qpos_head=[{values}]"
    values = ", ".join(f"{x:.5f}" for x in qpos)
    return f" qpos=[{values}]"


def _body_to_jsonable(body: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "position_rhs": [float(v) for v in pos],
            "quat_wxyz_rhs": [float(v) for v in quat],
        }
        for name, (pos, quat) in body.items()
    }


def _save_xrobot_frame(
    *,
    save_dir: Path,
    robot: str,
    frame_index: int,
    frame: XRobotBodyFrame,
    qpos: np.ndarray | None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now().astimezone()
    filename = saved_at.strftime("%Y%m%d_%H%M%S_%f") + f"_{robot}_frame{frame_index:06d}.json"
    path = save_dir / filename
    payload = {
        "schema": "kitov_deploy.xrobot_frame.v1",
        "saved_at": saved_at.isoformat(),
        "robot": robot,
        "frame_index": frame_index,
        "xrobot_timestamp_ns": int(frame.timestamp_ns),
        "joint_count": len(frame.body),
        "raw_unity_poses": frame.raw_poses,
        "converted_body": _body_to_jsonable(frame.body),
        "qpos": None if qpos is None else [float(v) for v in qpos],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def main() -> int:
    args = _parse_args()
    period_s = 1.0 / max(float(args.hz), 1e-6)
    print_interval = max(float(args.print_every), 0.05)

    stop = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    streamer = XRobotBodyStreamer()
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

    state_lock = threading.Lock()
    latest_frame: XRobotBodyFrame | None = None
    latest_frame_index = 0
    latest_qpos: np.ndarray | None = None

    def _on_viewer_key(key: int) -> None:
        if key != ord("P"):
            return
        with state_lock:
            frame = latest_frame
            frame_index = latest_frame_index
            qpos_snapshot = None if latest_qpos is None else latest_qpos.copy()
        if frame is None:
            print("[xrobot_retarget] P pressed, but no XRobot body frame has been received yet")
            return
        path = _save_xrobot_frame(
            save_dir=args.save_dir,
            robot=args.robot,
            frame_index=frame_index,
            frame=frame,
            qpos=qpos_snapshot,
        )
        print(f"[xrobot_retarget] saved XRobot frame: {path}")

    viewer = None
    if args.viewer:
        viewer = retargeter.make_viewer(motion_fps=args.hz, keyboard_callback=_on_viewer_key)

    streamer.start()
    print(
        "[xrobot_retarget] started "
        f"robot={args.robot} gmr_robot={retargeter.config.gmr_robot} "
        f"qpos={retargeter.robot_qpos_size} dof={retargeter.robot_dof_size} "
        f"ik_config={retargeter.config.ik_config_path}"
    )

    start_time = time.monotonic()
    last_print = start_time
    last_loop = start_time
    retarget_count = 0
    missing_count = 0
    last_qpos: np.ndarray | None = None

    try:
        while not stop:
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break

            frame = streamer.read_body_frame()
            if frame is None:
                missing_count += 1
                if now - last_print >= print_interval:
                    print(
                        "[xrobot_retarget] waiting for XRobot body frame "
                        f"t={now - start_time:.1f}s frames={retarget_count} missing={missing_count}"
                    )
                    last_print = now
                time.sleep(period_s)
                continue

            qpos = retargeter.retarget(frame.body, offset_to_ground=args.offset_to_ground)
            last_qpos = qpos
            retarget_count += 1
            with state_lock:
                latest_frame = frame
                latest_frame_index = retarget_count
                latest_qpos = qpos.copy()

            if viewer is not None:
                if args.show_human:
                    human_motion = retargeter.prepare_debug_human_data(
                        frame.body,
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
            else:
                elapsed = time.monotonic() - last_loop
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)
                last_loop = time.monotonic()

            now = time.monotonic()
            if now - last_print >= print_interval:
                fps = retarget_count / max(now - start_time, 1e-6)
                suffix = _format_qpos(qpos, args.print_qpos, has_floating_base=retargeter.has_floating_base)
                print(
                    "[xrobot_retarget] "
                    f"t={now - start_time:.1f}s frames={retarget_count} "
                    f"missing={missing_count} avg_hz={fps:.1f}{suffix}"
                )
                last_print = now
    finally:
        if viewer is not None:
            viewer.close()
        streamer.close()
        if last_qpos is None:
            print("[xrobot_retarget] stopped before receiving a body frame")
        else:
            print(f"[xrobot_retarget] stopped frames={retarget_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
