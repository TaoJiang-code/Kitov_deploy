#!/usr/bin/env python3
"""Live XRobot/PICO -> GMR -> BUMI RGMT policy -> Noetix BUMI hardware."""

from __future__ import annotations

import argparse
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT, OnlineGMRRetargeter
from kitov_deploy.gmr_viewer_process import GMRViewerProcess
from kitov_deploy.hardware.bumi_noetix_bridge import (
    DEFAULT_BUMI_NOETIX_HARDWARE_CONFIG,
    BumiNoetixBridge,
    load_bumi_noetix_hardware_config,
)
from kitov_deploy.policy_runtime import RobotState
from kitov_deploy.rgmt_runtime import DEFAULT_RGMT_MODEL_DIR, BumiRGMTRuntime
from kitov_deploy.xrobot_relay import UdpXRobotBodyReceiver
from kitov_deploy.xrobot_stream import XRobotBodyStreamer


class _TerminalKeyPoller:
    def __init__(self) -> None:
        self.enabled = bool(sys.stdin.isatty())
        self._old_attrs: list[Any] | None = None

    def __enter__(self) -> "_TerminalKeyPoller":
        if self.enabled:
            self._old_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self.enabled and self._old_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_attrs)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)


class _JointRateLimiter:
    def __init__(self, *, max_velocity_rad_s: float, hz: float) -> None:
        self.max_step = max(float(max_velocity_rad_s), 0.0) / max(float(hz), 1e-6)
        self._last: np.ndarray | None = None

    def reset(self, current: np.ndarray | None = None) -> None:
        self._last = None if current is None else np.asarray(current, dtype=np.float32).reshape(-1).copy()

    def limit(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if self._last is None or self.max_step <= 0.0:
            self._last = target.copy()
            return target.copy()
        delta = np.clip(target - self._last, -self.max_step, self.max_step)
        self._last = (self._last + delta).astype(np.float32)
        return self._last.copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live XRobot retargeting and BUMI RGMT policy on real Noetix BUMI.")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_RGMT_MODEL_DIR, help="BUMI RGMT model directory.")
    parser.add_argument("--policy-path", type=Path, default=None, help="Override RGMT policy path, .onnx or .pt.")
    parser.add_argument("--rgmt-config", type=Path, default=None, help="Override RGMT JSON/YAML config.")
    parser.add_argument("--hardware-config", type=Path, default=DEFAULT_BUMI_NOETIX_HARDWARE_CONFIG)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds. 0 means run until Ctrl-C.")
    parser.add_argument("--print-every", type=float, default=1.0)
    parser.add_argument("--actual-human-height", type=float, default=None)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=5e-1, help="IK damping passed to GMR.")
    parser.add_argument("--ground-offset", type=float, default=0.0)
    parser.add_argument("--offset-to-ground", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--body-source",
        choices=["local", "udp"],
        default="local",
        help="Read PICO frames from the local SDK or from the UDP relay.",
    )
    parser.add_argument("--body-bind-address", default="0.0.0.0", help="UDP bind address when --body-source=udp.")
    parser.add_argument("--body-port", type=int, default=47001, help="UDP port when --body-source=udp.")
    parser.add_argument("--body-source-host", default=None, help="Only accept UDP frames from this x86 host.")
    parser.add_argument("--quiet-gmr", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--send", action="store_true", help="Actually send commands to Noetix BUMI. Without this, dry-run only.")
    parser.add_argument("--gmr-viewer", action="store_true", help="Open GMR reference viewer.")
    parser.add_argument("--show-human", action="store_true", help="Draw human targets in the GMR viewer.")
    parser.add_argument(
        "--reference-delay-frames",
        type=int,
        default=-1,
        help="Delay policy reference center. -1 uses rgmt_command_window_after.",
    )
    return parser.parse_args()


def _state_from_qpos(qpos: np.ndarray) -> RobotState:
    return RobotState(
        root_pos=np.asarray(qpos[:3], dtype=np.float32),
        root_quat_wxyz=np.asarray(qpos[3:7], dtype=np.float32),
        dof_pos=np.asarray(qpos[7:], dtype=np.float32),
    )


def _qpos_from_state(state: RobotState) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state.root_pos, dtype=np.float32).reshape(3),
            np.asarray(state.root_quat_wxyz, dtype=np.float32).reshape(4),
            np.asarray(state.dof_pos, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )


def _apply_reference_ground_correction(
    policy: BumiRGMTRuntime,
    reference_state: RobotState,
    *,
    clearance: float = 0.0,
) -> RobotState:
    min_geom_z = policy.reference_min_geom_z(reference_state)
    root_pos = np.asarray(reference_state.root_pos, dtype=np.float32).reshape(3).copy()
    root_pos[2] -= float(min_geom_z) - float(clearance)
    return RobotState(
        root_pos=root_pos,
        root_quat_wxyz=np.asarray(reference_state.root_quat_wxyz, dtype=np.float32).copy(),
        dof_pos=np.asarray(reference_state.dof_pos, dtype=np.float32).copy(),
    )


def _mode_line(mode: str) -> str:
    if mode == "policy":
        return "policy"
    if mode == "zero":
        return "zero"
    return "damping"


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

    streamer = (
        XRobotBodyStreamer()
        if args.body_source == "local"
        else UdpXRobotBodyReceiver(
            bind_address=args.body_bind_address,
            port=args.body_port,
            source_host=args.body_source_host,
        )
    )
    retargeter = OnlineGMRRetargeter(
        "bumi",
        gmr_root=args.gmr_root,
        actual_human_height=args.actual_human_height,
        solver=args.solver,
        damping=args.damping,
        verbose=not args.quiet_gmr,
    )
    retargeter.set_ground_offset(args.ground_offset)
    policy = BumiRGMTRuntime(
        model_dir=args.model_dir,
        config_path=args.rgmt_config,
        policy_path=args.policy_path,
        hz=args.hz,
        device=args.device,
    )
    reference_delay_frames = (
        policy.config.command_window_after
        if int(args.reference_delay_frames) < 0
        else max(int(args.reference_delay_frames), 0)
    )
    if reference_delay_frames > policy.config.command_window_after:
        raise ValueError(
            "--reference-delay-frames cannot be greater than rgmt_command_window_after "
            f"({policy.config.command_window_after})."
        )

    hardware_config = load_bumi_noetix_hardware_config(args.hardware_config)
    bridge: BumiNoetixBridge | None = None
    limiter = _JointRateLimiter(max_velocity_rad_s=hardware_config.max_velocity_rad_s, hz=args.hz)
    if args.send:
        bridge = BumiNoetixBridge(hardware_config)
        print("[xrobot_bumi_rgmt_policy_real] connecting to Noetix BUMI")
        bridge.connect()
        initial = bridge.wait_for_state(hardware_config.recv_timeout_s)
        if initial is None:
            raise RuntimeError("No BUMI hardware state received after connecting to Noetix SDK.")
        limiter.reset(initial.dof_pos)
        bridge.send_damping()
        print("[xrobot_bumi_rgmt_policy_real] hardware connected; default mode=damping")
    else:
        print("[xrobot_bumi_rgmt_policy_real] dry-run mode: not connecting to Noetix SDK and not sending motor commands")

    gmr_viewer = (
        GMRViewerProcess("bumi", gmr_root=args.gmr_root, hz=args.hz, show_human=args.show_human)
        if args.gmr_viewer
        else None
    )

    streamer.start()
    print(
        "[xrobot_bumi_rgmt_policy_real] started "
        f"policy={policy.config.model_path} rgmt_config={policy.config.config_path} "
        f"hardware_config={hardware_config.path} reference_delay_frames={reference_delay_frames} "
        f"body_source={args.body_source}"
        + (f" body_source_host={args.body_source_host}" if args.body_source == "udp" else "")
    )
    print("[xrobot_bumi_rgmt_policy_real] keys: p=damping, 0=rate-limited joint zero, 1=policy")

    mode = "damping"
    start_time = time.monotonic()
    last_print = start_time
    last_loop = start_time
    frames = 0
    warmup_frames = 0
    missing = 0
    sent = 0
    last_result = None
    policy_primed = False
    last_robot_state: RobotState | None = None
    last_body_time = 0.0
    hold_target: np.ndarray | None = None

    try:
        with _TerminalKeyPoller() as key_poller:
            if not key_poller.enabled:
                print("[xrobot_bumi_rgmt_policy_real] stdin is not a TTY; keyboard control is disabled")
            while not stop:
                key = key_poller.poll()
                if key in ("p", "P"):
                    if mode != "damping":
                        print("[xrobot_bumi_rgmt_policy_real] mode=damping")
                    mode = "damping"
                    last_result = None
                    if bridge is not None:
                        bridge.send_damping()
                        sent += 1
                elif key == "0":
                    if mode != "zero":
                        print("[xrobot_bumi_rgmt_policy_real] mode=zero")
                    mode = "zero"
                    last_result = None
                    if bridge is not None:
                        state = bridge.read_state()
                        if state is not None:
                            limiter.reset(state.dof_pos)
                elif key == "1":
                    if mode != "zero":
                        print("[xrobot_bumi_rgmt_policy_real] ignoring policy request; press 0 and enter zero mode first")
                    elif mode != "policy":
                        print("[xrobot_bumi_rgmt_policy_real] mode=policy")
                        mode = "policy"
                        if bridge is not None:
                            state = bridge.read_state()
                            if state is not None:
                                limiter.reset(state.dof_pos)

                now = time.monotonic()
                if args.duration > 0.0 and now - start_time >= args.duration:
                    break

                frame = streamer.read_body_frame()
                if frame is None:
                    missing += 1
                else:
                    last_body_time = now
                    qpos = retargeter.retarget(frame.body, offset_to_ground=args.offset_to_ground)
                    reference_state = _state_from_qpos(qpos)
                    if args.offset_to_ground:
                        reference_state = _apply_reference_ground_correction(policy, reference_state)
                        qpos = _qpos_from_state(reference_state)
                    policy.update_reference(reference_state)
                    if policy.can_step_from_buffer(delay_frames=reference_delay_frames):
                        policy_primed = True
                    else:
                        warmup_frames += 1
                    if gmr_viewer is not None:
                        human_motion = retargeter.scaled_human_data if args.show_human else None
                        gmr_viewer.publish(qpos, human_motion_data=human_motion)
                    frames += 1

                if bridge is not None:
                    last_robot_state = bridge.robot_state(default_root_pos=policy.robot_config.default_root_pos)
                elif policy.can_step_from_buffer(delay_frames=reference_delay_frames):
                    last_robot_state = policy.reference_state_from_buffer(delay_frames=reference_delay_frames)

                if mode == "damping":
                    if bridge is not None:
                        bridge.send_damping()
                        sent += 1
                elif mode == "zero":
                    zero_target = limiter.limit(np.zeros(policy.robot_config.num_dof, dtype=np.float32))
                    hold_target = zero_target.copy()
                    if bridge is not None:
                        bridge.send_positions(zero_target, kp=hardware_config.zero_kp, kd=hardware_config.zero_kd)
                        sent += 1
                elif mode == "policy":
                    if (
                        policy_primed
                        and last_robot_state is not None
                        and policy.can_step_from_buffer(delay_frames=reference_delay_frames)
                    ):
                        result = policy.step_from_buffer(
                            delay_frames=reference_delay_frames,
                            robot_state=last_robot_state,
                        )
                        last_result = result
                        q_target = limiter.limit(result.q_target)
                        hold_target = q_target.copy()
                        if bridge is not None:
                            bridge.send_positions(q_target, kp=hardware_config.policy_kp, kd=hardware_config.policy_kd)
                            sent += 1
                    elif hold_target is not None:
                        if bridge is not None:
                            bridge.send_positions(hold_target, kp=hardware_config.zero_kp, kd=hardware_config.zero_kd)
                            sent += 1
                    elif bridge is not None:
                        bridge.send_damping()
                        sent += 1

                elapsed = time.monotonic() - last_loop
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)
                last_loop = time.monotonic()

                now = time.monotonic()
                if now - last_print >= print_interval:
                    avg_hz = frames / max(now - start_time, 1e-6)
                    status = (
                        "[xrobot_bumi_rgmt_policy_real] "
                        f"t={now - start_time:.1f}s frames={frames} warmup={warmup_frames} missing={missing} "
                        f"avg_hz={avg_hz:.1f} mode={_mode_line(mode)} send={int(args.send)} sent={sent} "
                        f"ref_buffer={policy.reference_buffer_size}"
                    )
                    if last_robot_state is not None:
                        status += (
                            f" q_range=[{np.min(last_robot_state.dof_pos):.3f},{np.max(last_robot_state.dof_pos):.3f}]"
                            f" root_w={last_robot_state.root_quat_wxyz[0]:.3f}"
                        )
                    if args.debug and last_result is not None:
                        status += (
                            f" raw_absmax={np.max(np.abs(last_result.raw_action)):.3f}"
                            f" target_range=[{np.min(last_result.q_target):.3f},{np.max(last_result.q_target):.3f}]"
                        )
                    print(status)
                    last_print = now
    finally:
        streamer.close()
        if bridge is not None:
            try:
                bridge.send_damping()
            except Exception as exc:
                print(f"[xrobot_bumi_rgmt_policy_real] failed to send final damping command: {exc}")
        policy.close()
        if gmr_viewer is not None:
            gmr_viewer.close()
        print(f"[xrobot_bumi_rgmt_policy_real] stopped frames={frames} sent={sent}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
