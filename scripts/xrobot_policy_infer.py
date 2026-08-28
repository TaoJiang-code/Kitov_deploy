#!/usr/bin/env python3
"""Live XRobot/PICO -> GMR qpos -> Kitov policy ONNX inference."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT, OnlineGMRRetargeter
from kitov_deploy.gmr_viewer_process import GMRViewerProcess
from kitov_deploy.mujoco_policy_sim import MujocoPolicySim
from kitov_deploy.policy_runtime import DEFAULT_MODEL_ROOT, KitovPolicyRuntime, RobotState
from kitov_deploy.xrobot_stream import XRobotBodyStreamer


POLICY_ROBOT_CHOICES = ["bumi", "g1", "unitree_g1"]


class _QTargetCommandFilter:
    def __init__(
        self,
        joint_names: list[str],
        *,
        alpha: float,
        deadband: float,
        arm_alpha: float,
        arm_deadband: float,
    ) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.deadband = max(float(deadband), 0.0)
        self.arm_alpha = min(max(float(arm_alpha), 0.0), 1.0)
        self.arm_deadband = max(float(arm_deadband), 0.0)
        self._last: np.ndarray | None = None
        self._arm_mask = np.asarray(
            [
                any(token in name.lower() for token in ("arm", "shoulder", "elbow", "wrist"))
                for name in joint_names
            ],
            dtype=bool,
        )

    def update(self, q_target: np.ndarray) -> np.ndarray:
        current = np.asarray(q_target, dtype=np.float32).reshape(-1)
        if self._last is None:
            self._last = current.copy()
            return current.copy()

        alpha = np.full(current.shape, self.alpha, dtype=np.float32)
        deadband = np.full(current.shape, self.deadband, dtype=np.float32)
        alpha[self._arm_mask] = self.arm_alpha
        deadband[self._arm_mask] = self.arm_deadband

        previous = self._last
        delta = current - previous
        filtered = previous + alpha * delta
        filtered[np.abs(delta) <= deadband] = previous[np.abs(delta) <= deadband]
        self._last = filtered.astype(np.float32).copy()
        return self._last.copy()

    def reset(self) -> None:
        self._last = None


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live XRobot retargeting and Kitov policy inference.")
    parser.add_argument("--robot", choices=POLICY_ROBOT_CHOICES, default="bumi", help="Target robot alias.")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT, help="Root directory containing per-robot model folders.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Override model directory for this robot.")
    parser.add_argument("--hz", type=float, default=50.0, help="Loop frequency.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds. 0 means run until Ctrl-C.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument("--actual-human-height", type=float, default=None, help="Optional human height used to scale IK targets.")
    parser.add_argument("--solver", default="daqp", help="IK solver passed to mink/qpsolvers.")
    parser.add_argument("--damping", type=float, default=5e-1, help="IK damping passed to GMR.")
    parser.add_argument("--ground-offset", type=float, default=0.0, help="Subtract this z offset from all human targets.")
    parser.add_argument("--offset-to-ground", action="store_true", help="Shift each frame so the lowest foot target sits above ground.")
    parser.add_argument("--device", default="cpu", help="ONNX Runtime device: cpu or cuda.")
    parser.add_argument("--max-z-delta", type=float, default=0.75, help="Per-step clamp for z updates. <=0 disables.")
    parser.add_argument("--z-window-size", type=int, default=3, help="Discounted z smoothing window.")
    parser.add_argument("--z-gamma", type=float, default=0.8, help="Discount factor for z smoothing.")
    parser.add_argument("--no-freeze-static-z", action="store_true", help="Always recompute backward z even when retargeted pose is unchanged.")
    parser.add_argument("--debug", action="store_true", help="Print detailed policy/z/sim diagnostics.")
    parser.add_argument("--quiet-gmr", action="store_true", help="Suppress GMR model/body/dof listing.")
    parser.add_argument("--print-q-target", choices=["none", "head", "all"], default="none", help="How much q_target to print.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer and apply policy q_target in sim.")
    parser.add_argument("--gmr-viewer", action="store_true", help="Open a second GMR retargeting viewer for the reference qpos.")
    parser.add_argument("--show-human", action="store_true", help="Draw GMR human targets in the GMR retargeting viewer.")
    parser.add_argument(
        "--control-source",
        choices=["policy", "reference", "hold"],
        default="policy",
        help="MuJoCo sim command source: policy q_target, GMR reference dof_pos, or current hold.",
    )
    parser.add_argument("--sim-substeps", type=int, default=0, help="MuJoCo steps per policy step. 0 means derive from model timestep and --hz.")
    parser.add_argument("--sim-kp-scale", type=float, default=1.0, help="Scale configured MuJoCo PD kp values.")
    parser.add_argument("--sim-kd-scale", type=float, default=1.0, help="Scale configured MuJoCo PD kd values.")
    parser.add_argument("--start-policy-enabled", action="store_true", help="Start MuJoCo policy control immediately. Default viewer mode waits for P.")
    parser.add_argument("--q-target-alpha", type=float, default=1.0, help="EMA alpha for policy q_target commands before MuJoCo PD. 1 disables smoothing.")
    parser.add_argument("--q-target-deadband", type=float, default=0.0, help="Hold q_target command joints when per-step change is below this value.")
    parser.add_argument("--arm-q-target-alpha", type=float, default=1.0, help="EMA alpha override for arm/shoulder/elbow/wrist q_target commands.")
    parser.add_argument("--arm-q-target-deadband", type=float, default=0.0, help="Deadband override for arm/shoulder/elbow/wrist q_target commands.")
    parser.add_argument("--no-floor", action="store_true", help="Do not inject floor/light assets into the policy MuJoCo viewer XML.")
    parser.add_argument("--elastic-band", action="store_true", help="Apply UFO-style elastic-band root support in MuJoCo.")
    parser.add_argument("--elastic-body", default=None, help="Body name for elastic band. Defaults to torso_link on G1, base_link on BUMI.")
    parser.add_argument("--elastic-stiffness", type=float, default=200.0)
    parser.add_argument("--elastic-damping", type=float, default=100.0)
    parser.add_argument("--elastic-length", type=float, default=0.0)
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
    policy: KitovPolicyRuntime,
    reference_state: RobotState,
    *,
    clearance: float = 0.0,
) -> tuple[RobotState, float]:
    min_geom_z = policy.reference_min_geom_z(reference_state)
    root_pos = np.asarray(reference_state.root_pos, dtype=np.float32).reshape(3).copy()
    root_pos[2] -= float(min_geom_z) - float(clearance)
    corrected = RobotState(
        root_pos=root_pos,
        root_quat_wxyz=np.asarray(reference_state.root_quat_wxyz, dtype=np.float32).copy(),
        dof_pos=np.asarray(reference_state.dof_pos, dtype=np.float32).copy(),
        root_lin_vel=reference_state.root_lin_vel,
        root_ang_vel=reference_state.root_ang_vel,
        dof_vel=reference_state.dof_vel,
    )
    return corrected, policy.reference_min_geom_z(corrected)


def _format_q_target(q_target: np.ndarray, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "head":
        values = ", ".join(f"{x:.4f}" for x in q_target[: min(8, q_target.size)])
        return f" q_target_head=[{values}]"
    values = ", ".join(f"{x:.5f}" for x in q_target)
    return f" q_target=[{values}]"


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
        verbose=not args.quiet_gmr,
    )
    retargeter.set_ground_offset(args.ground_offset)
    policy = KitovPolicyRuntime(
        args.robot,
        model_root=args.model_root,
        model_dir=args.model_dir,
        hz=args.hz,
        device=args.device,
        gamma=args.z_gamma,
        window_size=args.z_window_size,
        freeze_static_reference_z=not args.no_freeze_static_z,
        max_z_delta=args.max_z_delta,
    )
    q_target_filter = _QTargetCommandFilter(
        policy.robot_config.control_joint_names,
        alpha=args.q_target_alpha,
        deadband=args.q_target_deadband,
        arm_alpha=args.arm_q_target_alpha,
        arm_deadband=args.arm_q_target_deadband,
    )
    sim = (
        MujocoPolicySim(
            policy.robot_config,
            hz=args.hz,
            sim_substeps=args.sim_substeps,
            viewer=args.viewer,
            add_floor=not args.no_floor,
            kp_scale=args.sim_kp_scale,
            kd_scale=args.sim_kd_scale,
            elastic_band=args.elastic_band,
            elastic_body_name=args.elastic_body,
            elastic_stiffness=args.elastic_stiffness,
            elastic_damping=args.elastic_damping,
            elastic_length=args.elastic_length,
            start_policy_enabled=args.start_policy_enabled,
        )
        if args.viewer
        else None
    )
    gmr_viewer = (
        GMRViewerProcess(
            args.robot,
            gmr_root=args.gmr_root,
            hz=args.hz,
            show_human=args.show_human,
        )
        if args.gmr_viewer
        else None
    )

    streamer.start()
    print(
        "[xrobot_policy_infer] started "
        f"robot={args.robot} qpos={retargeter.robot_qpos_size} "
        f"policy_model={policy.bundle.policy_onnx} backward={policy.bundle.backward_onnx}"
        + (f" mujoco_xml={policy.robot_config.xml_path} sim_substeps={sim.sim_substeps}" if sim else "")
        + (" gmr_viewer=on" if gmr_viewer is not None else "")
    )
    if sim is not None and not sim.is_policy_enabled():
        print("[xrobot_policy_infer] policy MuJoCo control is paused; press p in this terminal or P in the policy viewer to toggle policy/damping.")

    start_time = time.monotonic()
    last_print = start_time
    last_loop = start_time
    frames = 0
    missing = 0
    sim_initialized = False
    policy_primed = False
    warmup_frames = 0
    sim_status = None
    last_result = None
    last_z = policy.current_z()
    was_policy_enabled = bool(sim.is_policy_enabled()) if sim is not None else True
    last_reference_state = None
    last_ref_min_geom_z = float("nan")

    try:
        with _TerminalKeyPoller() as key_poller:
            while not stop:
                key = key_poller.poll()
                if key in ("p", "P") and sim is not None:
                    sim.toggle_policy_control()
                    if not sim.is_policy_enabled():
                        last_result = None
                    else:
                        q_target_filter.reset()

                if sim is not None and not sim.is_running():
                    break

                now = time.monotonic()
                if args.duration > 0.0 and now - start_time >= args.duration:
                    break

                frame = streamer.read_body_frame()
                if frame is None:
                    missing += 1
                    if sim is not None:
                        robot_state = sim.robot_state()
                        if sim.is_policy_enabled() and last_reference_state is not None and policy.has_valid_reference_z:
                            if not was_policy_enabled:
                                q_target_filter.reset()
                            result = policy.step_with_z(last_z, robot_state=robot_state)
                            last_result = result
                            last_z = result.z
                            if args.control_source == "policy":
                                sim_q_target = q_target_filter.update(result.q_target)
                            elif args.control_source == "reference":
                                sim_q_target = last_reference_state.dof_pos
                            else:
                                sim_q_target = robot_state.dof_pos
                            sim_status = sim.step(sim_q_target)
                        else:
                            last_result = None
                            sim_status = sim.step_damping()
                        was_policy_enabled = sim.is_policy_enabled()
                    time.sleep(period_s)
                    continue

                qpos = retargeter.retarget(frame.body, offset_to_ground=args.offset_to_ground)
                reference_state = _state_from_qpos(qpos)
                if args.offset_to_ground:
                    reference_state, last_ref_min_geom_z = _apply_reference_ground_correction(policy, reference_state)
                    qpos = _qpos_from_state(reference_state)
                else:
                    last_ref_min_geom_z = policy.reference_min_geom_z(reference_state)
                last_reference_state = reference_state
                if not policy_primed:
                    if sim is not None and not sim_initialized:
                        sim.reset(reference_state)
                        sim_initialized = True
                    policy.prime_reference(reference_state)
                    if gmr_viewer is not None:
                        human_motion = retargeter.scaled_human_data if args.show_human else None
                        gmr_viewer.publish(qpos, human_motion_data=human_motion)
                    policy_primed = True
                    warmup_frames += 1
                    elapsed = time.monotonic() - last_loop
                    if elapsed < period_s:
                        time.sleep(period_s - elapsed)
                    last_loop = time.monotonic()
                    continue

                if sim is not None:
                    if not sim_initialized:
                        sim.reset(reference_state)
                        sim_initialized = True
                    robot_state = sim.robot_state()
                else:
                    robot_state = reference_state

                if sim is not None:
                    if not sim.is_policy_enabled():
                        last_z = policy.update_reference_z(reference_state)
                        last_result = None
                        sim_status = sim.step_damping()
                    else:
                        if not was_policy_enabled:
                            q_target_filter.reset()
                        result = policy.step(reference_state=reference_state, robot_state=robot_state)
                        last_result = result
                        last_z = result.z
                        if args.control_source == "policy":
                            sim_q_target = q_target_filter.update(result.q_target)
                        elif args.control_source == "reference":
                            sim_q_target = reference_state.dof_pos
                        else:
                            sim_q_target = robot_state.dof_pos
                        sim_status = sim.step(sim_q_target)
                    was_policy_enabled = sim.is_policy_enabled()
                else:
                    result = policy.step(reference_state=reference_state, robot_state=robot_state)
                    last_result = result
                    last_z = result.z
                if gmr_viewer is not None:
                    human_motion = retargeter.scaled_human_data if args.show_human else None
                    gmr_viewer.publish(qpos, human_motion_data=human_motion)
                frames += 1

                elapsed = time.monotonic() - last_loop
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)
                last_loop = time.monotonic()

                now = time.monotonic()
                if now - last_print >= print_interval:
                    fps = frames / max(now - start_time, 1e-6)
                    status = (
                        "[xrobot_policy_infer] "
                        f"t={now - start_time:.1f}s frames={frames} missing={missing} avg_hz={fps:.1f} "
                        f"control={args.control_source} "
                        + (f"sim_gate={'policy' if sim.is_policy_enabled() else 'damping'} " if sim is not None else "")
                        + f"z_norm={np.linalg.norm(last_z):.3f}"
                        + (
                            ""
                            if last_reference_state is None
                            else f" ref_z={last_reference_state.root_pos[2]:.3f}"
                        )
                        + (f" sim_z={sim_status.root_pos[2]:.3f}" if sim_status is not None else "")
                    )
                    if args.debug:
                        status += (
                            f" warmup={warmup_frames}"
                            + f" z_valid={int(policy.has_valid_reference_z)}"
                            + (
                                ""
                                if last_reference_state is None
                                else f" ref_min_geom_z={last_ref_min_geom_z:.3f}"
                            )
                        )
                        if last_result is not None:
                            status += (
                                f" raw_absmax={np.max(np.abs(last_result.raw_action)):.3f}"
                                + f" act_absmax={np.max(np.abs(last_result.normalized_action)):.3f}"
                                + f" q_range=[{np.min(last_result.q_target):.3f},{np.max(last_result.q_target):.3f}]"
                            )
                    if last_result is not None:
                        status += _format_q_target(last_result.q_target, args.print_q_target)
                    print(
                        status
                    )
                    last_print = now
    finally:
        streamer.close()
        if sim is not None:
            sim.close()
        if gmr_viewer is not None:
            gmr_viewer.close()
        print(f"[xrobot_policy_infer] stopped frames={frames}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
