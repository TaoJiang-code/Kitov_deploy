#!/usr/bin/env python3
"""Probe live XRobot/PICO body and controller data.

This script intentionally does not import GMR, MuJoCo, or policy code. It only
checks that xrobotoolkit_sdk can provide fresh body frames and controller state.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


XR_BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]


@dataclass
class ProbeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    callback_count: int = 0
    body_frame_count: int = 0
    last_body_recv_monotonic: float | None = None
    last_body_timestamp_ns: int | None = None
    last_top_timestamp_ns: int | None = None
    last_body_available: bool = False
    last_poses: Any = None
    last_buttons: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None


def _load_xrobot_sdk() -> Any:
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise SystemExit(
            "Failed to import xrobotoolkit_sdk. Activate the teleop/XRobot Python "
            "environment before running this probe."
        ) from exc

    if not hasattr(xrt, "init"):
        raise SystemExit("Installed xrobotoolkit_sdk does not expose init().")
    return xrt


def _has_callback_api(xrt: Any) -> bool:
    return all(
        hasattr(xrt, name)
        for name in ("register_frame_callback", "clear_frame_callback", "has_frame_callback")
    )


def _has_polling_api(xrt: Any) -> bool:
    return all(
        hasattr(xrt, name)
        for name in (
            "is_body_data_available",
            "get_body_joints_pose",
            "get_body_timestamp_ns",
            "get_A_button",
            "get_B_button",
            "get_X_button",
            "get_Y_button",
        )
    )


def _safe_xrt_call(xrt: Any, name: str, default: Any = None) -> Any:
    fn = getattr(xrt, name, None)
    if fn is None:
        return default
    try:
        return fn()
    except Exception:
        return default


def _axis(values: Any) -> list[float]:
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        try:
            return [float(values[0]), float(values[1])]
        except Exception:
            return [0.0, 0.0]
    return [0.0, 0.0]


def _polling_snapshot(xrt: Any) -> dict[str, Any]:
    body_available = bool(_safe_xrt_call(xrt, "is_body_data_available", False))
    body_timestamp_ns = int(_safe_xrt_call(xrt, "get_body_timestamp_ns", 0) or 0)
    top_timestamp_ns = int(_safe_xrt_call(xrt, "get_time_stamp_ns", body_timestamp_ns) or body_timestamp_ns)
    poses = _safe_xrt_call(xrt, "get_body_joints_pose", None) if body_available else None

    return {
        "timestamp_ns": top_timestamp_ns,
        "controllers": {
            "left": {
                "primary_button": bool(_safe_xrt_call(xrt, "get_X_button", False)),
                "secondary_button": bool(_safe_xrt_call(xrt, "get_Y_button", False)),
                "axis_click": bool(_safe_xrt_call(xrt, "get_left_axis_click", False)),
                "trigger": float(_safe_xrt_call(xrt, "get_left_trigger", 0.0) or 0.0),
                "grip": float(_safe_xrt_call(xrt, "get_left_grip", 0.0) or 0.0),
                "axis": _axis(_safe_xrt_call(xrt, "get_left_axis", [0.0, 0.0])),
            },
            "right": {
                "primary_button": bool(_safe_xrt_call(xrt, "get_A_button", False)),
                "secondary_button": bool(_safe_xrt_call(xrt, "get_B_button", False)),
                "axis_click": bool(_safe_xrt_call(xrt, "get_right_axis_click", False)),
                "trigger": float(_safe_xrt_call(xrt, "get_right_trigger", 0.0) or 0.0),
                "grip": float(_safe_xrt_call(xrt, "get_right_grip", 0.0) or 0.0),
                "axis": _axis(_safe_xrt_call(xrt, "get_right_axis", [0.0, 0.0])),
            },
        },
        "body": {
            "available": body_available,
            "timestamp_ns": body_timestamp_ns,
            "poses": poses,
        },
    }


def _buttons_from_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    controllers = snapshot.get("controllers", {})
    left = controllers.get("left", {}) if isinstance(controllers, dict) else {}
    right = controllers.get("right", {}) if isinstance(controllers, dict) else {}
    return {
        "left_key_one": bool(left.get("primary_button", False)),
        "left_key_two": bool(left.get("secondary_button", False)),
        "left_axis_click": bool(left.get("axis_click", False)),
        "left_index_trig": float(left.get("trigger", 0.0) or 0.0) > 1e-4,
        "left_grip": float(left.get("grip", 0.0) or 0.0) > 1e-4,
        "left_axis": _axis(left.get("axis", [0.0, 0.0])),
        "right_key_one": bool(right.get("primary_button", False)),
        "right_key_two": bool(right.get("secondary_button", False)),
        "right_axis_click": bool(right.get("axis_click", False)),
        "right_index_trig": float(right.get("trigger", 0.0) or 0.0) > 1e-4,
        "right_grip": float(right.get("grip", 0.0) or 0.0) > 1e-4,
        "right_axis": _axis(right.get("axis", [0.0, 0.0])),
    }


def _ingest_snapshot(state: ProbeState, snapshot: Any) -> None:
    now = time.monotonic()
    try:
        top_timestamp_ns = None
        body_timestamp_ns = None
        body_available = False
        poses = None

        if isinstance(snapshot, dict):
            raw_top = snapshot.get("timestamp_ns", None)
            top_timestamp_ns = int(raw_top) if raw_top not in (None, "") else None
            body = snapshot.get("body", {})
            if isinstance(body, dict):
                body_available = bool(body.get("available", False))
                raw_body_ts = body.get("timestamp_ns", None)
                body_timestamp_ns = int(raw_body_ts) if raw_body_ts not in (None, "") else None
                poses = body.get("poses", None)

        buttons = _buttons_from_snapshot(snapshot)
        with state.lock:
            state.callback_count += 1
            state.last_top_timestamp_ns = top_timestamp_ns
            state.last_body_available = body_available
            state.last_buttons = buttons
            if body_available:
                state.body_frame_count += 1
                state.last_body_recv_monotonic = now
                state.last_body_timestamp_ns = body_timestamp_ns
                state.last_poses = poses
            state.last_error = None
    except Exception as exc:
        with state.lock:
            state.last_error = str(exc)


def _pose_at(poses: Any, joint_name: str) -> tuple[list[float], list[float]] | None:
    if not isinstance(poses, (list, tuple)):
        return None
    try:
        idx = XR_BODY_JOINT_NAMES.index(joint_name)
    except ValueError:
        return None
    if idx >= len(poses):
        return None
    pose = poses[idx]
    if not isinstance(pose, (list, tuple)) or len(pose) < 7:
        return None
    try:
        x, y, z, qx, qy, qz, qw = [float(v) for v in pose[:7]]
    except Exception:
        return None
    return [x, y, z], [qx, qy, qz, qw]


def _fmt_vec(values: Iterable[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _print_status(state: ProbeState, selected_joints: list[str], start_time: float, prev_counts: tuple[int, int, float]) -> tuple[int, int, float]:
    now = time.monotonic()
    prev_callback_count, prev_body_count, prev_time = prev_counts
    with state.lock:
        callback_count = state.callback_count
        body_frame_count = state.body_frame_count
        last_body_recv = state.last_body_recv_monotonic
        last_body_ts = state.last_body_timestamp_ns
        last_top_ts = state.last_top_timestamp_ns
        body_available = state.last_body_available
        poses = state.last_poses
        buttons = dict(state.last_buttons)
        last_error = state.last_error

    elapsed = max(now - prev_time, 1e-6)
    cb_hz = (callback_count - prev_callback_count) / elapsed
    body_hz = (body_frame_count - prev_body_count) / elapsed
    age_ms = None if last_body_recv is None else (now - last_body_recv) * 1000.0
    pose_count = len(poses) if isinstance(poses, (list, tuple)) else 0

    print(
        "status "
        f"t={now - start_time:.1f}s "
        f"body_available={body_available} "
        f"poses={pose_count} "
        f"body_age_ms={'None' if age_ms is None else f'{age_ms:.1f}'} "
        f"body_ts_ns={last_body_ts} "
        f"top_ts_ns={last_top_ts} "
        f"callback_hz={cb_hz:.1f} "
        f"body_hz={body_hz:.1f}"
    )
    if last_error:
        print(f"  last_error={last_error}")

    for joint_name in selected_joints:
        pose = _pose_at(poses, joint_name)
        if pose is None:
            print(f"  {joint_name}: unavailable")
            continue
        pos, quat_xyzw = pose
        print(f"  {joint_name}: pos={_fmt_vec(pos)} quat_xyzw={_fmt_vec(quat_xyzw, precision=4)}")

    if buttons:
        print(
            "  buttons "
            f"A={buttons.get('right_key_one')} "
            f"B={buttons.get('right_key_two')} "
            f"X={buttons.get('left_key_one')} "
            f"Y={buttons.get('left_key_two')} "
            f"L_trig={buttons.get('left_index_trig')} "
            f"R_trig={buttons.get('right_index_trig')}"
        )

    return callback_count, body_frame_count, now


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read live XRobot/PICO body data without retargeting.")
    parser.add_argument("--hz", type=float, default=50.0, help="Polling rate when callback API is unavailable.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after this many seconds. 0 means run until Ctrl-C.")
    parser.add_argument(
        "--mode",
        choices=["auto", "callback", "polling"],
        default="auto",
        help="SDK read mode. auto prefers callback when available.",
    )
    parser.add_argument(
        "--joints",
        type=str,
        default="Pelvis,Left_Foot,Right_Foot,Head,Left_Wrist,Right_Wrist",
        help="Comma-separated joint names to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    xrt = _load_xrobot_sdk()

    callback_ok = _has_callback_api(xrt)
    polling_ok = _has_polling_api(xrt)
    if args.mode == "callback" and not callback_ok:
        raise SystemExit("Requested --mode callback, but this xrobotoolkit_sdk has no callback API.")
    if args.mode == "polling" and not polling_ok:
        raise SystemExit("Requested --mode polling, but this xrobotoolkit_sdk has no polling API.")
    if args.mode == "auto":
        if callback_ok:
            mode = "callback"
        elif polling_ok:
            mode = "polling"
        else:
            raise SystemExit("xrobotoolkit_sdk exposes neither supported callback nor polling APIs.")
    else:
        mode = args.mode

    selected_joints = [x.strip() for x in str(args.joints).split(",") if x.strip()]
    unknown_joints = [name for name in selected_joints if name not in XR_BODY_JOINT_NAMES]
    if unknown_joints:
        raise SystemExit(f"Unknown joint(s): {unknown_joints}. Known joints: {XR_BODY_JOINT_NAMES}")

    stop_event = threading.Event()

    def _request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("[xrobot_probe] initializing xrobotoolkit_sdk")
    xrt.init()
    print(
        "[xrobot_probe] started "
        f"mode={mode} callback_api={callback_ok} polling_api={polling_ok} "
        f"hz={float(args.hz):.1f} print_every={float(args.print_every):.2f}s"
    )

    state = ProbeState()
    poll_thread: threading.Thread | None = None

    if mode == "callback":
        xrt.register_frame_callback(lambda snapshot: _ingest_snapshot(state, snapshot))
    else:
        period_s = 1.0 / max(float(args.hz), 1e-6)

        def _poll_loop() -> None:
            while not stop_event.is_set():
                _ingest_snapshot(state, _polling_snapshot(xrt))
                stop_event.wait(timeout=period_s)

        poll_thread = threading.Thread(target=_poll_loop, name="xrobot-probe-poll", daemon=True)
        poll_thread.start()

    start_time = time.monotonic()
    prev_counts = (0, 0, start_time)
    print_interval = max(float(args.print_every), 0.05)
    duration = max(float(args.duration), 0.0)

    try:
        while not stop_event.is_set():
            now = time.monotonic()
            if duration > 0.0 and now - start_time >= duration:
                break
            prev_counts = _print_status(state, selected_joints, start_time, prev_counts)
            stop_event.wait(timeout=print_interval)
    finally:
        stop_event.set()
        if mode == "callback" and hasattr(xrt, "clear_frame_callback"):
            try:
                xrt.clear_frame_callback()
            except Exception:
                pass
        if poll_thread is not None:
            poll_thread.join(timeout=1.0)
        if hasattr(xrt, "close"):
            try:
                xrt.close()
            except Exception:
                pass
        print("[xrobot_probe] stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
