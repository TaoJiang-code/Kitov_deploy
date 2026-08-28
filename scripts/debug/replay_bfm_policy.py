#!/usr/bin/env python3
"""Replay a BFM motion through Kitov policy inference and MuJoCo sim."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT
from kitov_deploy.gmr_viewer_process import GMRViewerProcess
from kitov_deploy.mujoco_policy_sim import MujocoPolicySim
from kitov_deploy.policy_runtime import (
    DEFAULT_MODEL_ROOT,
    KitovPolicyRuntime,
    RobotState,
    _compute_humanoid_observations_max_np,
    _quat_rotate_inverse_xyzw,
    _quat_to_ang_vel_xyzw,
)


DEFAULT_BUMI_DATA = Path("/home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/bumi/bumi_lafan_full.pkl")
DEFAULT_G1_DATA = Path("/home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/g1/lafan_29dof.pkl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one BFM motion through Kitov policy sim.")
    parser.add_argument("--robot", choices=["bumi", "g1", "unitree_g1"], default="bumi")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--motion-index", type=int, default=None)
    parser.add_argument("--motion-key", default=None)
    parser.add_argument("--list-motions", action="store_true", help="Print motion keys and exit.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--viewer", action="store_true", help="Open policy MuJoCo viewer.")
    parser.add_argument("--reference-viewer", action="store_true", help="Open a second viewer for the reference motion.")
    parser.add_argument(
        "--control-source",
        choices=["policy", "reference", "hold"],
        default="policy",
        help="MuJoCo sim command source.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-every", type=float, default=1.0)
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--elastic-band", action="store_true", help="Apply UFO-style elastic-band root support in MuJoCo.")
    parser.add_argument("--elastic-body", default=None, help="Body name for elastic band. Defaults to torso_link on G1, base_link on BUMI.")
    parser.add_argument("--elastic-stiffness", type=float, default=200.0)
    parser.add_argument("--elastic-damping", type=float, default=100.0)
    parser.add_argument("--elastic-length", type=float, default=0.0)
    return parser.parse_args()


def _default_data_path(robot: str) -> Path:
    return DEFAULT_G1_DATA if robot in {"g1", "unitree_g1"} else DEFAULT_BUMI_DATA


def _default_model_dir(robot: str) -> Path:
    return Path("models/g1/kitov_fb_g1") if robot in {"g1", "unitree_g1"} else Path("models/bumi/kitov_fb_bumi_action_scale_0.5")


def _load_joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import joblib. Install it in the Kitov_deploy conda environment first:\n"
            "python -m pip install joblib"
        ) from exc
    return joblib


def _resample_motion(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    *,
    source_fps: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_fps = float(source_fps)
    target_hz = float(target_hz)
    if root_pos.shape[0] <= 1 or abs(source_fps - target_hz) < 1e-6:
        return root_pos, root_quat_xyzw, dof_pos

    from scipy.spatial.transform import Rotation, Slerp

    src_t = np.arange(root_pos.shape[0], dtype=np.float64) / source_fps
    dst_count = int(np.floor(src_t[-1] * target_hz)) + 1
    dst_t = np.arange(dst_count, dtype=np.float64) / target_hz
    dst_t = np.clip(dst_t, src_t[0], src_t[-1])
    root_pos_out = np.stack([np.interp(dst_t, src_t, root_pos[:, i]) for i in range(3)], axis=-1)
    dof_out = np.stack([np.interp(dst_t, src_t, dof_pos[:, i]) for i in range(dof_pos.shape[1])], axis=-1)
    slerp = Slerp(src_t, Rotation.from_quat(root_quat_xyzw))
    quat_out = slerp(dst_t).as_quat()
    return root_pos_out.astype(np.float32), quat_out.astype(np.float32), dof_out.astype(np.float32)


def _select_motion_key(keys: list[str], args: argparse.Namespace) -> str:
    if args.motion_key is not None:
        return str(args.motion_key)
    if args.motion_index is not None:
        return keys[int(args.motion_index)]
    if args.robot in {"g1", "unitree_g1"}:
        blocked = ("fall", "getup", "get_up", "lie")
        for key in keys:
            lowered = key.lower()
            if not any(token in lowered for token in blocked):
                return key
    return keys[0]


def _load_motion(args: argparse.Namespace) -> tuple[str, float, np.ndarray, np.ndarray, np.ndarray]:
    joblib = _load_joblib()
    data_path = args.data_path or _default_data_path(args.robot)
    data = joblib.load(data_path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty motion dict in {args.data_path}")
    keys = list(data.keys())
    if args.list_motions:
        for idx, key in enumerate(keys):
            print(f"{idx}: {key}")
        raise SystemExit(0)
    key = _select_motion_key(keys, args)
    if key not in data:
        raise KeyError(f"Motion key {key!r} not found. First keys: {keys[:10]}")
    motion = data[key]
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    if "root_quat" in motion:
        root_quat_xyzw = np.asarray(motion["root_quat"], dtype=np.float32)
    elif "root_rot" in motion:
        root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    else:
        raise ValueError(f"Motion {key!r} is missing root_quat/root_rot")
    if "dof_pos" in motion:
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    elif "dof" in motion:
        dof_pos = np.asarray(motion["dof"], dtype=np.float32)
    else:
        raise ValueError(f"Motion {key!r} is missing dof_pos/dof")
    fps = float(motion.get("fps", args.hz))
    root_pos, root_quat_xyzw, dof_pos = _resample_motion(
        root_pos,
        root_quat_xyzw,
        dof_pos,
        source_fps=fps,
        target_hz=args.hz,
    )
    return str(key), fps, root_pos, root_quat_xyzw, dof_pos


def _state_at(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    idx: int,
    qvel: np.ndarray | None = None,
) -> RobotState:
    quat_xyzw = root_quat_xyzw[idx]
    qvel_frame = None if qvel is None else np.asarray(qvel[idx], dtype=np.float32)
    return RobotState(
        root_pos=root_pos[idx].astype(np.float32),
        root_quat_wxyz=quat_xyzw[[3, 0, 1, 2]].astype(np.float32),
        dof_pos=dof_pos[idx].astype(np.float32),
        root_lin_vel=None if qvel_frame is None else qvel_frame[:3],
        root_ang_vel=None if qvel_frame is None else qvel_frame[3:6],
        dof_vel=None if qvel_frame is None else qvel_frame[6:],
    )


def _expert_qpos_qvel(
    policy: KitovPolicyRuntime,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    *,
    hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    frame_count = int(dof_pos.shape[0])
    qpos = np.zeros((frame_count, policy.fk.model.nq), dtype=np.float64)
    qvel = np.zeros((frame_count, policy.fk.model.nv), dtype=np.float64)
    qpos[:, :3] = root_pos.astype(np.float64)
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float64)
    qpos[:, 7:] = dof_pos.astype(np.float64)
    dt = 1.0 / max(float(hz), 1e-6)
    if frame_count > 1:
        for idx in range(frame_count - 1):
            mujoco.mj_differentiatePos(policy.fk.model, qvel[idx], dt, qpos[idx], qpos[idx + 1])
        qvel[-1] = qvel[-2]
    return qpos, qvel


def _precompute_expert_z(
    policy: KitovPolicyRuntime,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    hz: float,
) -> np.ndarray:
    frame_count = int(dof_pos.shape[0])
    if frame_count <= 0:
        raise ValueError("Cannot precompute z for an empty motion")
    import mujoco

    body_pos_rows: list[np.ndarray] = []
    body_rot_rows: list[np.ndarray] = []
    body_vel_rows: list[np.ndarray] = []
    body_ang_vel_rows: list[np.ndarray] = []
    for idx in range(frame_count):
        policy.fk.data.qpos[:] = qpos[idx]
        policy.fk.data.qvel[:] = qvel[idx]
        mujoco.mj_forward(policy.fk.model, policy.fk.data)
        mujoco.mj_fwdVelocity(policy.fk.model, policy.fk.data)
        state = _state_at(root_pos, root_quat_xyzw, dof_pos, idx, qvel)
        body_pos, body_rot = policy.fk.bodies_from_state(state)
        body_vel = np.zeros_like(body_pos, dtype=np.float32)
        body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)
        base_body_count = len(policy.fk.body_ids)
        for out_idx, body_id in enumerate(policy.fk.body_ids):
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(policy.fk.model, policy.fk.data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0)
            body_ang_vel[out_idx] = velocity[:3].astype(np.float32)
            body_vel[out_idx] = velocity[3:].astype(np.float32)
        if body_pos.shape[0] > base_body_count and idx > 0:
            body_vel[base_body_count:] = (body_pos[base_body_count:] - body_pos_rows[-1][base_body_count:]) * float(hz)
            body_ang_vel[base_body_count:] = _quat_to_ang_vel_xyzw(
                body_rot_rows[-1][base_body_count:],
                body_rot[base_body_count:],
                1.0 / max(float(hz), 1e-6),
            ).astype(np.float32)
        body_pos_rows.append(body_pos.astype(np.float32))
        body_rot_rows.append(body_rot.astype(np.float32))
        body_vel_rows.append(body_vel.astype(np.float32))
        body_ang_vel_rows.append(body_ang_vel.astype(np.float32))
    body_pos = np.stack(body_pos_rows, axis=0)
    body_rot = np.stack(body_rot_rows, axis=0)
    body_vel = np.stack(body_vel_rows, axis=0)
    body_ang_vel = np.stack(body_ang_vel_rows, axis=0)
    if frame_count > 1:
        body_vel[0, len(policy.fk.body_ids):] = body_vel[1, len(policy.fk.body_ids):]
        body_ang_vel[0, len(policy.fk.body_ids):] = body_ang_vel[1, len(policy.fk.body_ids):]
    dof_vel = qvel[:, 6:].astype(np.float32)

    obs_dict = _compute_humanoid_observations_max_np(
        body_pos.astype(np.float32),
        body_rot.astype(np.float32),
        body_vel.astype(np.float32),
        body_ang_vel.astype(np.float32),
        local_root_obs=True,
        root_height_obs=policy.robot_config.root_height_obs,
    )
    privileged_state = np.concatenate([v.astype(np.float32) for v in obs_dict.values()], axis=-1)
    projected_gravity = _quat_rotate_inverse_xyzw(
        body_rot[:, 0, :].astype(np.float32),
        np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (frame_count, 1)),
    )
    ref_dof_pos = (dof_pos.astype(np.float32) - policy.robot_config.default_joint_angles[None, :]).astype(np.float32)
    ref_ang_vel = (body_ang_vel[:, 0, :].astype(np.float32) * policy.robot_config.base_ang_vel_scale).astype(np.float32)
    state = np.concatenate([ref_dof_pos, dof_vel.astype(np.float32), projected_gravity, ref_ang_vel], axis=-1).astype(np.float32)

    feed: dict[str, np.ndarray] = {}
    if "state" in policy.backward_inputs:
        feed["state"] = state
    if "last_action" in policy.backward_inputs:
        feed["last_action"] = np.zeros((frame_count, policy.robot_config.num_dof), dtype=np.float32)
    if "privileged_state" in policy.backward_inputs:
        feed["privileged_state"] = privileged_state
    z = np.asarray(policy.backward_session.run([policy.backward_output], feed)[0], dtype=np.float32)
    if z.shape != (frame_count, policy.z_dim):
        raise ValueError(f"Precomputed z shape mismatch: expected {(frame_count, policy.z_dim)}, got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise RuntimeError("Precomputed z contains non-finite values")
    return z


def main() -> int:
    args = _parse_args()
    stop = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    motion_key, source_fps, root_pos, root_quat_xyzw, dof_pos = _load_motion(args)
    model_dir = args.model_dir or _default_model_dir(args.robot)
    policy = KitovPolicyRuntime(
        args.robot,
        model_root=args.model_root,
        model_dir=model_dir,
        hz=args.hz,
        device=args.device,
    )
    if dof_pos.shape[1] != policy.robot_config.num_dof:
        raise ValueError(
            f"Motion dof width {dof_pos.shape[1]} does not match robot num_dof={policy.robot_config.num_dof}"
        )
    qpos_seq, qvel_seq = _expert_qpos_qvel(policy, root_pos, root_quat_xyzw, dof_pos, hz=args.hz)
    z_seq = _precompute_expert_z(policy, root_pos, root_quat_xyzw, dof_pos, qpos_seq, qvel_seq, hz=args.hz)

    sim = MujocoPolicySim(
        policy.robot_config,
        hz=args.hz,
        viewer=args.viewer,
        add_floor=not args.no_floor,
        elastic_band=args.elastic_band,
        elastic_body_name=args.elastic_body,
        elastic_stiffness=args.elastic_stiffness,
        elastic_damping=args.elastic_damping,
        elastic_length=args.elastic_length,
    )
    reference_viewer = (
        GMRViewerProcess(args.robot, gmr_root=args.gmr_root, hz=args.hz, show_human=False)
        if args.reference_viewer
        else None
    )

    frame_count = min(root_pos.shape[0], int(args.max_frames) if args.max_frames > 0 else root_pos.shape[0])
    first_state = _state_at(root_pos, root_quat_xyzw, dof_pos, 0, qvel_seq)
    sim.reset(first_state)

    print(
        "[replay_bfm_policy] started "
        f"robot={args.robot} motion={motion_key!r} source_fps={source_fps:g} replay_hz={args.hz:g} "
        f"frames={frame_count}/{root_pos.shape[0]} control={args.control_source} "
        f"model={policy.bundle.policy_onnx}"
    )

    start = time.monotonic()
    last_print = start
    raw_absmax_values: list[float] = []
    act_absmax_values: list[float] = []
    dof_mae_values: list[float] = []
    root_err_values: list[float] = []
    last_status = None

    try:
        for frame_idx in range(frame_count):
            if stop or (args.viewer and not sim.is_running()):
                break
            loop_start = time.monotonic()
            reference_state = _state_at(root_pos, root_quat_xyzw, dof_pos, frame_idx, qvel_seq)
            robot_state = sim.robot_state()
            result = policy.step_with_z(z_seq[frame_idx], robot_state=robot_state)

            if args.control_source == "policy":
                sim_q_target = result.q_target
            elif args.control_source == "reference":
                sim_q_target = reference_state.dof_pos
            else:
                sim_q_target = robot_state.dof_pos
            last_status = sim.step(sim_q_target)

            if reference_viewer is not None:
                reference_qpos = np.concatenate(
                    [reference_state.root_pos, reference_state.root_quat_wxyz, reference_state.dof_pos],
                    axis=0,
                )
                reference_viewer.publish(reference_qpos)

            raw_absmax_values.append(float(np.max(np.abs(result.raw_action))))
            act_absmax_values.append(float(np.max(np.abs(result.normalized_action))))
            dof_mae_values.append(float(np.mean(np.abs(last_status.dof_pos - reference_state.dof_pos))))
            root_err_values.append(float(np.linalg.norm(last_status.root_pos - reference_state.root_pos)))

            now = time.monotonic()
            if now - last_print >= max(float(args.print_every), 0.05):
                print(
                    "[replay_bfm_policy] "
                    f"frame={frame_idx + 1}/{frame_count} "
                    f"raw_absmax={raw_absmax_values[-1]:.3f} act_absmax={act_absmax_values[-1]:.3f} "
                    f"q_range=[{np.min(result.q_target):.3f},{np.max(result.q_target):.3f}] "
                    f"dof_mae={dof_mae_values[-1]:.3f} root_err={root_err_values[-1]:.3f} "
                    f"sim_z={last_status.root_pos[2]:.3f}"
                )
                last_print = now

            if args.viewer:
                elapsed = time.monotonic() - loop_start
                period = 1.0 / max(float(args.hz), 1e-6)
                if elapsed < period:
                    time.sleep(period - elapsed)
    finally:
        if reference_viewer is not None:
            reference_viewer.close()
        sim.close()

    if raw_absmax_values:
        print(
            "[replay_bfm_policy] summary "
            f"frames={len(raw_absmax_values)} "
            f"raw_absmax_mean={np.mean(raw_absmax_values):.3f} raw_absmax_max={np.max(raw_absmax_values):.3f} "
            f"act_absmax_mean={np.mean(act_absmax_values):.3f} act_absmax_max={np.max(act_absmax_values):.3f} "
            f"dof_mae_mean={np.mean(dof_mae_values):.3f} dof_mae_max={np.max(dof_mae_values):.3f} "
            f"root_err_mean={np.mean(root_err_values):.3f} root_err_max={np.max(root_err_values):.3f}"
        )
    else:
        print("[replay_bfm_policy] stopped before running any frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
