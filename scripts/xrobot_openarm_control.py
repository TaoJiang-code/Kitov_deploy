#!/usr/bin/env python3
"""Live XRobot/PICO -> local GMR -> OpenArm CAN targets.

Default mode is dry-run: it computes and prints hardware targets but never
imports openarm_can or sends CAN frames. Real hardware control requires
``--send --enable-motors``.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT, OnlineGMRRetargeter  # noqa: E402
from kitov_deploy.hardware.openarm_can_bridge import (  # noqa: E402
    DEFAULT_OPENARM_HARDWARE_CONFIG,
    OpenArmCANBridge,
    OpenArmCommandLimiter,
    OpenArmQposMapper,
    load_openarm_hardware_config,
)
from kitov_deploy.xrobot_stream import XRobotBodyStreamer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live XRobot retargeting to OpenArm v1 CAN targets.")
    parser.add_argument("--hardware-config", type=Path, default=DEFAULT_OPENARM_HARDWARE_CONFIG)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--hz", type=float, default=50.0, help="Retarget/control rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds. 0 means run until Ctrl-C.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first XRobot body frame before enabling motors. 0 means wait forever.",
    )
    parser.add_argument("--actual-human-height", type=float, default=None)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=5e-1)
    parser.add_argument("--ground-offset", type=float, default=0.0)
    parser.add_argument("--offset-to-ground", action="store_true")
    parser.add_argument("--use-velocity-limit", action="store_true")
    parser.add_argument("--quiet-gmr", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Open OpenArm MuJoCo retarget viewer.")
    parser.add_argument("--show-human", action="store_true")
    parser.add_argument("--show-all-human", action="store_true")
    parser.add_argument("--human-axes-only", action="store_true")
    parser.add_argument("--show-human-name", action="store_true")
    parser.add_argument("--send", action="store_true", help="Send MIT position targets to openarm_can.")
    parser.add_argument("--enable-motors", action="store_true", help="Call enable_all before sending commands.")
    parser.add_argument("--disable-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--print-targets", choices=["none", "head", "all"], default="head")
    return parser.parse_args()


def _format_targets(targets: dict[str, float], mode: str) -> str:
    if mode == "none":
        return ""
    items = sorted(targets.items())
    if mode == "head":
        items = items[: min(6, len(items))]
    joined = ", ".join(f"{name}={value:.4f}" for name, value in items)
    return f" targets[{joined}]"


def _state_targets(states: dict[str, Any]) -> dict[str, float]:
    return {name: float(state.position) for name, state in states.items()}


def main() -> int:
    args = _parse_args()
    if args.send and not args.enable_motors:
        raise SystemExit("--send requires --enable-motors. Use dry-run without --send until mapping is verified.")

    hardware_config = load_openarm_hardware_config(args.hardware_config)
    mapper = OpenArmQposMapper(hardware_config)
    limiter = OpenArmCommandLimiter(hardware_config)

    stop = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    bridge = None
    motors_enabled = False
    if args.send:
        bridge = OpenArmCANBridge(hardware_config)
        print("[xrobot_openarm_control] connecting to OpenArm CAN")
        bridge.connect()
        try:
            limiter.reset(_state_targets(bridge.read_state()))
            print("[xrobot_openarm_control] initialized rate limiter from current motor positions")
        except Exception as exc:
            raise SystemExit(
                "Cannot read initial motor state. Refusing to send because rate limiting "
                f"needs the real starting joint positions: {exc}"
            ) from exc
        print("[xrobot_openarm_control] waiting for first XRobot body frame before enabling motors")
    else:
        neutral_qpos = np.zeros(mapper.model.nq, dtype=np.float64)
        limiter.reset(mapper.hardware_targets_from_qpos(neutral_qpos))
        print("[xrobot_openarm_control] dry-run mode: not importing openarm_can and not sending CAN frames")

    streamer = XRobotBodyStreamer()
    retargeter = OnlineGMRRetargeter(
        "openarm_v1",
        gmr_root=args.gmr_root,
        actual_human_height=args.actual_human_height,
        solver=args.solver,
        damping=args.damping,
        use_velocity_limit=args.use_velocity_limit,
        verbose=not args.quiet_gmr,
    )
    retargeter.set_ground_offset(args.ground_offset)

    viewer = None
    if args.viewer:
        viewer = retargeter.make_viewer(motion_fps=args.hz)

    period_s = 1.0 / max(float(args.hz), 1e-6)
    print_interval = max(float(args.print_every), 0.05)
    watchdog_timeout_s = max(float(hardware_config.safety.watchdog_timeout_s), 0.0)
    startup_timeout_s = max(float(args.startup_timeout), 0.0)
    start_time = time.monotonic()
    last_body_time: float | None = None
    last_print = 0.0
    last_loop = start_time
    frames = 0
    missing = 0
    sent = 0

    streamer.start()
    print(
        "[xrobot_openarm_control] started "
        f"qpos={retargeter.robot_qpos_size} dof={retargeter.robot_dof_size} "
        f"hardware_config={args.hardware_config}"
    )

    try:
        while not stop:
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break

            frame = streamer.read_body_frame()
            if frame is None:
                missing += 1
                if frames == 0:
                    if startup_timeout_s > 0.0 and now - start_time > startup_timeout_s:
                        print(
                            "[xrobot_openarm_control] no XRobot body frame before startup timeout; "
                            "motors were not enabled"
                        )
                        break
                    if now - last_print >= print_interval:
                        print(
                            "[xrobot_openarm_control] "
                            f"waiting for first XRobot body frame t={now - start_time:.1f}s missing={missing}"
                        )
                        last_print = now
                    time.sleep(period_s)
                    continue
                assert last_body_time is not None
                if args.send and watchdog_timeout_s > 0.0 and now - last_body_time > watchdog_timeout_s:
                    print(
                        "[xrobot_openarm_control] XRobot body frame watchdog timeout; "
                        "disabling motors and stopping"
                    )
                    assert bridge is not None
                    if motors_enabled:
                        bridge.disable_all()
                    break
                time.sleep(period_s)
                continue

            last_body_time = now
            qpos = retargeter.retarget(frame.body, offset_to_ground=args.offset_to_ground)
            raw_targets = mapper.hardware_targets_from_qpos(qpos)
            dt = max(now - last_loop, period_s)
            targets = limiter.limit(raw_targets, dt)
            command_qpos = mapper.qpos_from_hardware_targets(qpos, targets)
            frames += 1

            if args.send:
                assert bridge is not None
                if not motors_enabled:
                    bridge.enable_all()
                    motors_enabled = True
                    print("[xrobot_openarm_control] motors enabled after first XRobot body frame")
                bridge.send_position_targets(targets, kp_scale=args.kp_scale, kd_scale=args.kd_scale)
                sent += 1

            if viewer is not None:
                human_motion = None
                if args.show_human:
                    human_motion = retargeter.prepare_debug_human_data(
                        frame.body,
                        offset_to_ground=args.offset_to_ground,
                        include_unscaled=args.show_all_human,
                    )
                viewer.step_qpos(
                    command_qpos,
                    human_motion_data=human_motion,
                    show_human_body_name=args.show_human_name,
                    show_human_points=not args.human_axes_only,
                    rate_limit=True,
                )
            else:
                elapsed = time.monotonic() - now
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)

            last_loop = time.monotonic()
            now = last_loop
            if now - last_print >= print_interval:
                avg_hz = frames / max(now - start_time, 1e-6)
                print(
                    "[xrobot_openarm_control] "
                    f"t={now - start_time:.1f}s frames={frames} missing={missing} "
                    f"sent={sent} avg_hz={avg_hz:.1f}"
                    f"{_format_targets(targets, args.print_targets)}"
                )
                last_print = now
    finally:
        if viewer is not None:
            viewer.close()
        streamer.close()
        if bridge is not None and args.disable_on_exit and motors_enabled:
            bridge.disable_all()
        print(f"[xrobot_openarm_control] stopped frames={frames} sent={sent}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
