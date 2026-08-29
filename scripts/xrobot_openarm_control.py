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
from typing import Any, Callable

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
    parser.add_argument(
        "--startup-hold-warmup",
        type=float,
        default=1.0,
        help="Seconds to wait for stable motor state before sending the first hold target.",
    )
    parser.add_argument(
        "--startup-position-abs-limit",
        type=float,
        default=float(2.0 * np.pi),
        help="Reject startup hold targets whose absolute motor position exceeds this value. 0 disables the check.",
    )
    parser.add_argument(
        "--startup-limit-margin",
        type=float,
        default=0.05,
        help="Startup range tolerance in radians before clamping tiny out-of-range motor readings.",
    )
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


def _format_limits(limits: dict[str, tuple[float | None, float | None]], mode: str) -> str:
    if mode == "none":
        return ""
    items = sorted(limits.items())
    if mode == "head":
        items = items[: min(6, len(items))]
    joined = ", ".join(
        f"{name}=[{('-inf' if lower is None else f'{lower:.3f}')},"
        f"{('inf' if upper is None else f'{upper:.3f}')}]"
        for name, (lower, upper) in items
    )
    return f" limits[{joined}]"


def _state_targets(states: dict[str, Any]) -> dict[str, float]:
    return {name: float(state.position) for name, state in states.items()}


def _normalize_target_to_limits(
    value: float,
    lower: float | None,
    upper: float | None,
    *,
    margin_rad: float,
) -> tuple[float, str | None]:
    if lower is None and upper is None:
        return value, None

    margin = max(float(margin_rad), 0.0)
    candidates = [float(value)]
    two_pi = float(2.0 * np.pi)
    candidates.extend(float(value) + two_pi * k for k in range(-4, 5) if k != 0)

    def _within(candidate: float) -> bool:
        if lower is not None and candidate < lower - margin:
            return False
        if upper is not None and candidate > upper + margin:
            return False
        return True

    valid_candidates = [candidate for candidate in candidates if _within(candidate)]
    if not valid_candidates:
        return value, None

    if lower is not None and upper is not None:
        center = 0.5 * (lower + upper)
        normalized = min(valid_candidates, key=lambda candidate: abs(candidate - center))
    else:
        normalized = min(valid_candidates, key=abs)

    clipped = normalized
    if lower is not None:
        clipped = max(clipped, lower)
    if upper is not None:
        clipped = min(clipped, upper)

    if abs(clipped - value) > 1e-9:
        return float(clipped), f"{value:.4f}->{clipped:.4f}"
    return float(clipped), None


def _normalize_hold_targets(
    targets: dict[str, float],
    hardware_limits: dict[str, tuple[float | None, float | None]],
    *,
    margin_rad: float,
) -> tuple[dict[str, float], list[str]]:
    normalized = dict(targets)
    changes: list[str] = []
    for name, value in sorted(targets.items()):
        lower, upper = hardware_limits.get(name, (None, None))
        new_value, change = _normalize_target_to_limits(
            float(value),
            lower,
            upper,
            margin_rad=margin_rad,
        )
        normalized[name] = new_value
        if change is not None:
            changes.append(f"{name}:{change}")
    return normalized, changes


def _validate_hold_targets(
    targets: dict[str, float],
    expected_names: set[str],
    *,
    hardware_limits: dict[str, tuple[float | None, float | None]],
    abs_limit_rad: float,
    margin_rad: float,
) -> tuple[bool, str]:
    missing = sorted(expected_names.difference(targets))
    if missing:
        return False, f"missing={missing[:4]}"
    for name in sorted(expected_names):
        value = float(targets[name])
        if not np.isfinite(value):
            return False, f"{name} is not finite: {value}"
        lower, upper = hardware_limits.get(name, (None, None))
        if lower is not None and value < lower - margin_rad:
            return False, f"{name}={value:.4f} below hardware lower {lower:.4f}"
        if upper is not None and value > upper + margin_rad:
            return False, f"{name}={value:.4f} above hardware upper {upper:.4f}"
        if abs_limit_rad > 0.0 and abs(value) > abs_limit_rad:
            return False, f"{name}={value:.4f} exceeds startup limit {abs_limit_rad:.4f}"
    return True, "ok"


def _read_startup_hold_targets(
    bridge: OpenArmCANBridge,
    hardware_config: Any,
    *,
    hardware_limits: dict[str, tuple[float | None, float | None]],
    warmup_s: float,
    abs_limit_rad: float,
    margin_rad: float,
    sample_period_s: float,
    print_interval_s: float,
    should_stop: Callable[[], bool],
) -> dict[str, float] | None:
    expected_names = {motor.joint_name for motor in hardware_config.motors}
    start = time.monotonic()
    last_print = 0.0
    attempts = 0
    last_reason = "not read yet"

    while not should_stop():
        attempts += 1
        states = bridge.read_state(recv_timeout_us=hardware_config.safety.enable_recv_timeout_us)
        targets = _state_targets(states)
        targets, changes = _normalize_hold_targets(targets, hardware_limits, margin_rad=margin_rad)
        ok, reason = _validate_hold_targets(
            targets,
            expected_names,
            hardware_limits=hardware_limits,
            abs_limit_rad=abs_limit_rad,
            margin_rad=margin_rad,
        )
        elapsed = time.monotonic() - start
        if ok and elapsed >= warmup_s:
            if changes:
                print(
                    "[xrobot_openarm_control] normalized startup motor state "
                    f"{', '.join(changes[:8])}"
                )
            return targets

        last_reason = "warming up" if ok else reason
        now = time.monotonic()
        if now - last_print >= print_interval_s:
            print(
                "[xrobot_openarm_control] "
                f"waiting for valid startup motor state t={elapsed:.1f}s attempts={attempts} "
                f"reason={last_reason}"
            )
            last_print = now
        time.sleep(max(float(sample_period_s), 1e-6))
    return None


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

    period_s = 1.0 / max(float(args.hz), 1e-6)
    print_interval = max(float(args.print_every), 0.05)

    bridge = None
    motors_enabled = False
    initial_targets: dict[str, float] | None = None
    hardware_limits = mapper.hardware_limits()
    if args.send:
        bridge = OpenArmCANBridge(hardware_config)
        print("[xrobot_openarm_control] connecting to OpenArm CAN")
        bridge.connect()
        bridge.enable_all()
        motors_enabled = True
        print(
            "[xrobot_openarm_control] motors enabled; reading startup hold posture"
            f"{_format_limits(hardware_limits, args.print_targets)}"
        )
        try:
            initial_targets = _read_startup_hold_targets(
                bridge,
                hardware_config,
                hardware_limits=hardware_limits,
                warmup_s=max(float(args.startup_hold_warmup), 0.0),
                abs_limit_rad=max(float(args.startup_position_abs_limit), 0.0),
                margin_rad=max(float(args.startup_limit_margin), 0.0),
                sample_period_s=period_s,
                print_interval_s=print_interval,
                should_stop=lambda: stop,
            )
        except Exception as exc:
            if motors_enabled:
                bridge.disable_all()
            raise SystemExit(
                "Cannot read a safe startup motor state. Refusing to send hold targets: "
                f"{exc}"
            ) from exc
        if initial_targets is None:
            if motors_enabled:
                bridge.disable_all()
            print("[xrobot_openarm_control] stopped before startup hold posture was available")
            return 130
        limiter.reset(initial_targets)
        bridge.send_position_targets(initial_targets, kp_scale=args.kp_scale, kd_scale=args.kd_scale)
        print(
            "[xrobot_openarm_control] initialized hold target from validated motor state; "
            "holding current posture until first XRobot body frame"
            f"{_format_targets(initial_targets, args.print_targets)}"
        )
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

    start_time = time.monotonic()
    last_print = 0.0
    last_loop = start_time
    last_targets: dict[str, float] | None = None if initial_targets is None else dict(initial_targets)
    last_command_qpos: np.ndarray | None = None
    last_human_motion: Any | None = None
    frames = 0
    missing = 0
    held = 0
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

                if args.send and last_targets is not None:
                    assert bridge is not None
                    bridge.send_position_targets(last_targets, kp_scale=args.kp_scale, kd_scale=args.kd_scale)
                    sent += 1
                held += 1
                if viewer is not None and last_command_qpos is not None:
                    viewer.step_qpos(
                        last_command_qpos,
                        human_motion_data=last_human_motion,
                        show_human_body_name=args.show_human_name,
                        show_human_points=not args.human_axes_only,
                        rate_limit=True,
                    )
                else:
                    time.sleep(period_s)
                last_loop = time.monotonic()
                if now - last_print >= print_interval:
                    avg_hz = frames / max(now - start_time, 1e-6)
                    state = "waiting for first XRobot body frame" if frames == 0 else "holding last target"
                    print(
                        "[xrobot_openarm_control] "
                        f"{state} t={now - start_time:.1f}s frames={frames} "
                        f"missing={missing} held={held} sent={sent} avg_hz={avg_hz:.1f}"
                        f"{_format_targets(last_targets or {}, args.print_targets)}"
                    )
                    last_print = now
                continue

            qpos = retargeter.retarget(frame.body, offset_to_ground=args.offset_to_ground)
            raw_targets = mapper.hardware_targets_from_qpos(qpos)
            dt = max(now - last_loop, period_s)
            targets = limiter.limit(raw_targets, dt)
            command_qpos = mapper.qpos_from_hardware_targets(qpos, targets)
            human_motion = None
            if args.show_human:
                human_motion = retargeter.prepare_debug_human_data(
                    frame.body,
                    offset_to_ground=args.offset_to_ground,
                    include_unscaled=args.show_all_human,
                )
            last_targets = dict(targets)
            last_command_qpos = command_qpos.copy()
            last_human_motion = human_motion
            frames += 1

            if args.send:
                assert bridge is not None
                bridge.send_position_targets(targets, kp_scale=args.kp_scale, kd_scale=args.kd_scale)
                sent += 1

            if viewer is not None:
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
                    f"held={held} sent={sent} avg_hz={avg_hz:.1f}"
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
