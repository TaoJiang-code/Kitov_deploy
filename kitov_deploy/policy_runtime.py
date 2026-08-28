"""ONNX inference runtime for Kitov tracking policies."""

from __future__ import annotations

import json
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kitov_deploy.mjcf_utils import prepared_mjcf_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = REPO_ROOT / "models"


ROBOT_ALIASES = {
    "g1": "g1",
    "unitree_g1": "g1",
    "bumi": "bumi",
}


@dataclass(frozen=True)
class RobotPolicyConfig:
    robot_name: str
    xml_path: Path
    default_root_pos: np.ndarray
    default_root_quat_wxyz: np.ndarray
    root_height_obs: bool
    base_ang_vel_scale: float
    normalize_action_to: float
    action_clip_value: float
    control_joint_names: list[str]
    default_joint_angles: np.ndarray
    action_target_scale: np.ndarray
    joint_pos_lower_limit: np.ndarray
    joint_pos_upper_limit: np.ndarray
    joint_velocity_limit: np.ndarray
    q_target_slew_safety_factor: float
    sim_joint_kp: np.ndarray
    sim_joint_kd: np.ndarray
    sim_effort_limit: np.ndarray
    sim_joint_armature: np.ndarray
    sim_joint_frictionloss: np.ndarray
    sim_timestep: float | None
    camera_body_name: str | None
    body_names: list[str]
    extend_config: list[dict[str, Any]]

    @property
    def num_dof(self) -> int:
        return len(self.control_joint_names)


@dataclass(frozen=True)
class ModelBundle:
    model_dir: Path
    policy_onnx: Path
    policy_meta: Path
    backward_onnx: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RobotState:
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray
    root_lin_vel: np.ndarray | None = None
    root_ang_vel: np.ndarray | None = None
    dof_vel: np.ndarray | None = None


@dataclass(frozen=True)
class PolicyStepResult:
    z: np.ndarray
    actor_obs: np.ndarray
    raw_action: np.ndarray
    normalized_action: np.ndarray
    q_target: np.ndarray


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import onnxruntime. Install it in the Kitov_deploy conda "
            "environment first:\npython -m pip install onnxruntime"
        ) from exc
    return ort


def resolve_robot_name(robot: str) -> str:
    key = robot.lower()
    if key not in ROBOT_ALIASES:
        supported = ", ".join(sorted(ROBOT_ALIASES))
        raise ValueError(f"Unsupported robot '{robot}'. Supported robots: {supported}")
    return ROBOT_ALIASES[key]


def load_robot_policy_config(robot: str, config_path: Path | None = None) -> RobotPolicyConfig:
    robot_name = resolve_robot_name(robot)
    path = config_path or (REPO_ROOT / "configs" / "policy" / f"{robot_name}.json")
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))

    control_joint_names = [str(name) for name in payload["control_joint_names"]]
    default_joint_angles = np.asarray(payload["default_joint_angles"], dtype=np.float32)
    action_target_scale = np.asarray(payload["action_target_scale"], dtype=np.float32)
    if default_joint_angles.shape != (len(control_joint_names),):
        raise ValueError(f"{path}: default_joint_angles shape mismatch")
    if action_target_scale.shape != (len(control_joint_names),):
        raise ValueError(f"{path}: action_target_scale shape mismatch")

    def _array(name: str, default: list[float]) -> np.ndarray:
        value = np.asarray(payload.get(name, default), dtype=np.float32)
        if value.shape != (len(control_joint_names),):
            raise ValueError(f"{path}: {name} shape mismatch")
        return value

    default_root_pos = np.asarray(payload.get("default_root_pos", [0.0, 0.0, 0.8]), dtype=np.float32)
    default_root_quat_wxyz = np.asarray(payload.get("default_root_quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    if default_root_pos.shape != (3,):
        raise ValueError(f"{path}: default_root_pos shape mismatch")
    if default_root_quat_wxyz.shape != (4,):
        raise ValueError(f"{path}: default_root_quat_wxyz shape mismatch")

    xml_path = Path(str(payload["xml_path"])).expanduser()
    if not xml_path.is_absolute():
        xml_path = REPO_ROOT / xml_path

    return RobotPolicyConfig(
        robot_name=str(payload["robot_name"]),
        xml_path=xml_path,
        default_root_pos=default_root_pos,
        default_root_quat_wxyz=default_root_quat_wxyz,
        root_height_obs=bool(payload.get("root_height_obs", True)),
        base_ang_vel_scale=float(payload.get("base_ang_vel_scale", 0.25)),
        normalize_action_to=float(payload.get("normalize_action_to", 5.0)),
        action_clip_value=float(payload.get("action_clip_value", 5.0)),
        control_joint_names=control_joint_names,
        default_joint_angles=default_joint_angles,
        action_target_scale=action_target_scale,
        joint_pos_lower_limit=_array("joint_pos_lower_limit", [-np.inf] * len(control_joint_names)),
        joint_pos_upper_limit=_array("joint_pos_upper_limit", [np.inf] * len(control_joint_names)),
        joint_velocity_limit=_array("joint_velocity_limit", [np.inf] * len(control_joint_names)),
        q_target_slew_safety_factor=float(payload.get("q_target_slew_safety_factor", 0.5)),
        sim_joint_kp=_array("sim_joint_kp", [40.0] * len(control_joint_names)),
        sim_joint_kd=_array("sim_joint_kd", [1.0] * len(control_joint_names)),
        sim_effort_limit=_array("sim_effort_limit", [80.0] * len(control_joint_names)),
        sim_joint_armature=_array("sim_joint_armature", [0.0] * len(control_joint_names)),
        sim_joint_frictionloss=_array("sim_joint_frictionloss", [0.0] * len(control_joint_names)),
        sim_timestep=float(payload["sim_timestep"]) if payload.get("sim_timestep") is not None else None,
        camera_body_name=str(payload["camera_body_name"]) if payload.get("camera_body_name") else None,
        body_names=[str(name) for name in payload["body_names"]],
        extend_config=[dict(item) for item in payload.get("extend_config", [])],
    )


def resolve_model_bundle(
    robot: str,
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    model_dir: Path | str | None = None,
) -> ModelBundle:
    robot_name = resolve_robot_name(robot)
    model_root = Path(model_root).expanduser()
    root = Path(model_dir).expanduser() if model_dir is not None else model_root / robot_name
    candidates = [root / "exported", root]
    errors: list[str] = []

    for candidate in candidates:
        if not candidate.exists():
            errors.append(f"{candidate}: directory does not exist")
            continue
        backward = candidate / "backward_encoder.onnx"
        meta_files = sorted(path for path in candidate.glob("*.meta.json") if path.is_file())
        policy_pairs = []
        for meta in meta_files:
            onnx = candidate / meta.name.replace(".meta.json", ".onnx")
            if onnx.name == "backward_encoder.onnx":
                continue
            if onnx.exists():
                policy_pairs.append((onnx, meta))
        if backward.exists() and policy_pairs:
            policy_onnx, policy_meta = policy_pairs[0]
            metadata = json.loads(policy_meta.read_text(encoding="utf-8"))
            return ModelBundle(
                model_dir=candidate,
                policy_onnx=policy_onnx,
                policy_meta=policy_meta,
                backward_onnx=backward,
                metadata=metadata,
            )
        errors.append(f"{candidate}: expected backward_encoder.onnx and one *.meta.json + matching .onnx policy")

    tried = "\n".join(f"  - {error}" for error in errors)
    raise FileNotFoundError(
        f"Cannot find Kitov model bundle for robot={robot_name!r}.\n"
        "Copy exported model files into one of these layouts:\n"
        f"  {model_root / robot_name / 'exported'}/backward_encoder.onnx\n"
        f"  {model_root / robot_name / 'exported'}/FBcprAuxModel.onnx\n"
        f"  {model_root / robot_name / 'exported'}/FBcprAuxModel.meta.json\n"
        "or put the same files directly under the robot model directory.\n"
        f"Tried:\n{tried}"
    )


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return q[..., [1, 2, 3, 0]]


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q).copy()
    out[..., 0:3] *= -1.0
    return out


def _quat_normalize_xyzw(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[..., 3:4]
    q_vec = q[..., 0:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def _quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[..., 3:4]
    q_vec = q[..., 0:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def _quat_from_angle_axis_xyzw(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(angle)
    return np.concatenate([axis * np.sin(half)[..., None], np.cos(half)[..., None]], axis=-1)


def _calc_heading_quat_inv_xyzw(q: np.ndarray) -> np.ndarray:
    ref_dir = np.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1.0
    rot_dir = _quat_rotate_xyzw(q, ref_dir)
    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    axis = np.zeros_like(q[..., 0:3])
    axis[..., 2] = 1.0
    return _quat_from_angle_axis_xyzw(-heading, axis)


def _quat_to_tan_norm_xyzw(q: np.ndarray) -> np.ndarray:
    ref_tan = np.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1.0
    ref_norm = np.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1.0
    tan = _quat_rotate_xyzw(q, ref_tan)
    norm = _quat_rotate_xyzw(q, ref_norm)
    return np.concatenate([tan, norm], axis=-1)


def _quat_to_ang_vel_xyzw(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    q_prev = _quat_normalize_xyzw(q_prev)
    q_curr = _quat_normalize_xyzw(q_curr)
    dots = np.sum(q_prev * q_curr, axis=-1)
    q_curr = np.where((dots < 0.0)[..., None], -q_curr, q_curr)
    dq = _quat_mul_xyzw(_quat_conj_xyzw(q_prev), q_curr)
    dq = _quat_normalize_xyzw(dq)
    v = dq[..., 0:3]
    w = np.clip(dq[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arctan2(np.linalg.norm(v, axis=-1), w)
    axis = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)
    omega = (angle[..., None] / max(float(dt), 1e-6)) * axis
    tiny = angle < 1e-6
    if np.any(tiny):
        omega[tiny] = (2.0 / max(float(dt), 1e-6)) * v[tiny]
    return omega


def _compute_humanoid_observations_max_np(
    body_pos: np.ndarray,
    body_rot: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
    *,
    local_root_obs: bool,
    root_height_obs: bool,
) -> OrderedDict[str, np.ndarray]:
    obs_dict: OrderedDict[str, np.ndarray] = OrderedDict()
    t_count, body_count, _ = body_pos.shape
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    if root_height_obs:
        obs_dict["root_height"] = root_pos[:, 2:3]

    heading_rot_inv = _calc_heading_quat_inv_xyzw(root_rot)
    heading_rot_inv_expand = np.repeat(heading_rot_inv[:, None, :], body_count, axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(t_count * body_count, 4)

    local_body_pos = body_pos - root_pos[:, None, :]
    flat_local_body_pos = local_body_pos.reshape(t_count * body_count, 3)
    flat_local_body_pos = _quat_rotate_xyzw(flat_heading_rot_inv, flat_local_body_pos)
    obs_dict["local_body_pos"] = flat_local_body_pos.reshape(t_count, body_count * 3)[..., 3:]

    flat_body_rot = body_rot.reshape(t_count * body_count, 4)
    flat_local_body_rot = _quat_mul_xyzw(flat_heading_rot_inv, flat_body_rot)
    local_body_rot_obs = _quat_to_tan_norm_xyzw(flat_local_body_rot).reshape(t_count, body_count * 6)
    if not local_root_obs:
        local_body_rot_obs = local_body_rot_obs.copy()
        local_body_rot_obs[..., 0:6] = _quat_to_tan_norm_xyzw(root_rot)
    obs_dict["local_body_rot"] = local_body_rot_obs

    flat_body_vel = body_vel.reshape(t_count * body_count, 3)
    obs_dict["local_body_vel"] = _quat_rotate_xyzw(flat_heading_rot_inv, flat_body_vel).reshape(t_count, body_count * 3)

    flat_body_ang_vel = body_ang_vel.reshape(t_count * body_count, 3)
    obs_dict["local_body_ang_vel"] = _quat_rotate_xyzw(flat_heading_rot_inv, flat_body_ang_vel).reshape(t_count, body_count * 3)
    return obs_dict


class ForwardKinematics:
    def __init__(self, robot_config: RobotPolicyConfig) -> None:
        import mujoco

        if not robot_config.xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {robot_config.xml_path}")
        self.robot_config = robot_config
        with prepared_mjcf_path(robot_config.xml_path) as xml_path:
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 7 + robot_config.num_dof:
            raise ValueError(
                f"Robot XML nq={self.model.nq} does not match config num_dof={robot_config.num_dof}"
            )

        body_ids = []
        for name in robot_config.body_names:
            try:
                body_ids.append(self.model.body(name).id)
            except KeyError as exc:
                raise KeyError(f"Body {name!r} not found in {robot_config.xml_path}") from exc
        self.body_ids = np.asarray(body_ids, dtype=np.int32)

    def bodies_from_state(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[0:3] = np.asarray(state.root_pos, dtype=np.float64).reshape(3)
        qpos[3:7] = np.asarray(state.root_quat_wxyz, dtype=np.float64).reshape(4)
        qpos[7:] = np.asarray(state.dof_pos, dtype=np.float64).reshape(self.robot_config.num_dof)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        body_pos = self.data.xpos[self.body_ids].copy()
        body_rot_xyzw = _wxyz_to_xyzw(self.data.xquat[self.body_ids].copy())

        names = list(self.robot_config.body_names)
        for ext in self.robot_config.extend_config:
            parent_idx = names.index(str(ext["parent_name"]))
            pos_in_parent = np.asarray(ext["pos"], dtype=np.float64).reshape(3)
            rot_xyzw = _wxyz_to_xyzw(np.asarray(ext["rot"], dtype=np.float64).reshape(4))
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot_xyzw[parent_idx]
            ext_pos = parent_pos + _quat_rotate_xyzw(parent_rot[None, :], pos_in_parent[None, :])[0]
            ext_rot = _quat_mul_xyzw(parent_rot[None, :], rot_xyzw[None, :])[0]
            body_pos = np.concatenate([body_pos, ext_pos[None, :]], axis=0)
            body_rot_xyzw = np.concatenate([body_rot_xyzw, ext_rot[None, :]], axis=0)
            names.append(str(ext["joint_name"]))
        return body_pos.astype(np.float64), body_rot_xyzw.astype(np.float64)

    def min_geom_z_from_state(self, state: RobotState) -> float:
        import mujoco

        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[0:3] = np.asarray(state.root_pos, dtype=np.float64).reshape(3)
        qpos[3:7] = np.asarray(state.root_quat_wxyz, dtype=np.float64).reshape(4)
        qpos[7:] = np.asarray(state.dof_pos, dtype=np.float64).reshape(self.robot_config.num_dof)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return float(np.min(self.data.geom_xpos[:, 2]))


class KitovPolicyRuntime:
    def __init__(
        self,
        robot: str,
        *,
        model_root: Path = DEFAULT_MODEL_ROOT,
        model_dir: Path | None = None,
        robot_config_path: Path | None = None,
        hz: float = 50.0,
        device: str = "cpu",
        ctx_norm_ref: float = 16.0,
        gamma: float = 0.8,
        window_size: int = 3,
        drop_first_z_frames: int = 1,
        freeze_static_reference_z: bool = True,
        static_reference_tol: float = 1e-3,
        max_z_delta: float = 0.75,
    ) -> None:
        self.robot_config = load_robot_policy_config(robot, robot_config_path)
        self.bundle = resolve_model_bundle(robot, model_root=model_root, model_dir=model_dir)
        self._validate_metadata()

        ort = _load_onnxruntime()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if str(device).startswith("cuda") else ["CPUExecutionProvider"]
        self.backward_session = ort.InferenceSession(str(self.bundle.backward_onnx), providers=providers)
        self.policy_session = ort.InferenceSession(str(self.bundle.policy_onnx), providers=providers)
        self.backward_inputs = {item.name for item in self.backward_session.get_inputs()}
        self.backward_output = self.backward_session.get_outputs()[0].name
        self.policy_input = self.policy_session.get_inputs()[0].name
        self.policy_output = self.policy_session.get_outputs()[0].name

        self.fk = ForwardKinematics(self.robot_config)
        self.dt = 1.0 / max(float(hz), 1e-6)
        self.z_dim = int(self.bundle.metadata["z_dim"])
        self.ctx_norm_ref = float(ctx_norm_ref)
        self.gamma = float(gamma)
        self.window_size = max(int(window_size), 1)
        self.drop_first_z_frames = max(int(drop_first_z_frames), 0)
        self.freeze_static_reference_z = bool(freeze_static_reference_z)
        self.static_reference_tol = float(static_reference_tol)
        self.max_z_delta = float(max_z_delta)
        self.z_window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.last_z = self._default_z()
        self.last_action = np.zeros(self.robot_config.num_dof, dtype=np.float32)
        self.history_buffers: dict[str, deque[np.ndarray]] = {
            "actions": deque(maxlen=4),
            "base_ang_vel": deque(maxlen=4),
            "dof_pos": deque(maxlen=4),
            "dof_vel": deque(maxlen=4),
            "projected_gravity": deque(maxlen=4),
        }
        self.prev_reference_state: RobotState | None = None
        self.prev_reference_bodies: tuple[np.ndarray, np.ndarray] | None = None
        self.reference_z_step_count = 0
        self.has_valid_reference_z = False
        self.prev_robot_state: RobotState | None = None
        self.last_q_target: np.ndarray | None = None
        self.last_step_monotonic: float | None = None

    def _validate_metadata(self) -> None:
        meta = self.bundle.metadata
        num_dof = self.robot_config.num_dof
        if int(meta.get("num_dof", -1)) != num_dof:
            raise ValueError(f"Model num_dof={meta.get('num_dof')} does not match robot config num_dof={num_dof}")
        if int(meta.get("output_action_dim", -1)) != num_dof:
            raise ValueError("Model output_action_dim does not match robot config")
        model_joints = [str(name) for name in meta.get("control_joint_names", [])]
        if model_joints and model_joints != self.robot_config.control_joint_names:
            raise ValueError("Model control_joint_names do not match deploy robot config")

    def _default_z(self) -> np.ndarray:
        z = np.zeros(int(self.bundle.metadata["z_dim"]), dtype=np.float32)
        if z.size:
            z[0] = self.ctx_norm_ref
        return z

    def _validate_z(self, z: Any) -> np.ndarray | None:
        try:
            z_np = np.asarray(z, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if z_np.shape != (self.z_dim,) or not np.all(np.isfinite(z_np)):
            return None
        return z_np

    def _accept_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32).reshape(self.z_dim)
        if self.max_z_delta > 0.0 and self.last_z.shape == z.shape:
            z = (self.last_z + np.clip(z - self.last_z, -self.max_z_delta, self.max_z_delta)).astype(np.float32)
        self.last_z = z
        self.z_window.appendleft(z)
        return self._smooth_z()

    def _state_velocities(self, current: RobotState, previous: RobotState | None, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if current.dof_vel is not None:
            dof_vel = np.asarray(current.dof_vel, dtype=np.float32).reshape(self.robot_config.num_dof)
        elif previous is not None:
            dof_vel = (np.asarray(current.dof_pos, dtype=np.float32) - np.asarray(previous.dof_pos, dtype=np.float32)) / dt
        else:
            dof_vel = np.zeros(self.robot_config.num_dof, dtype=np.float32)

        if current.root_ang_vel is not None:
            root_ang_vel = np.asarray(current.root_ang_vel, dtype=np.float32).reshape(3)
        else:
            root_ang_vel = np.zeros(3, dtype=np.float32)
        return dof_vel.astype(np.float32), root_ang_vel.astype(np.float32)

    def prime_reference(self, reference_state: RobotState) -> None:
        """Record a reference frame for velocity differencing without running policy."""
        body_pos, body_rot = self.fk.bodies_from_state(reference_state)
        self.prev_reference_state = reference_state
        self.prev_reference_bodies = (body_pos, body_rot)
        self.reference_z_step_count = max(self.reference_z_step_count, self.drop_first_z_frames)
        self.has_valid_reference_z = False

    def _reference_z(self, reference_state: RobotState, dt: float) -> np.ndarray:
        body_pos, body_rot = self.fk.bodies_from_state(reference_state)
        if self.prev_reference_state is None or self.prev_reference_bodies is None:
            self.prev_reference_state = reference_state
            self.prev_reference_bodies = (body_pos, body_rot)
            self.reference_z_step_count += 1
            return self.last_z.copy()
        if (
            self.freeze_static_reference_z
            and self.has_valid_reference_z
            and self.reference_z_step_count >= self.drop_first_z_frames
            and np.allclose(reference_state.dof_pos, self.prev_reference_state.dof_pos, rtol=0.0, atol=self.static_reference_tol)
            and np.allclose(reference_state.root_pos, self.prev_reference_state.root_pos, rtol=0.0, atol=self.static_reference_tol)
            and np.allclose(reference_state.root_quat_wxyz, self.prev_reference_state.root_quat_wxyz, rtol=0.0, atol=self.static_reference_tol)
        ):
            self.reference_z_step_count += 1
            return self.last_z.copy()
        else:
            prev_body_pos, prev_body_rot = self.prev_reference_bodies
            body_vel = (body_pos - prev_body_pos) / dt
            body_ang_vel = _quat_to_ang_vel_xyzw(prev_body_rot, body_rot, dt)
            dof_vel = (
                np.asarray(reference_state.dof_pos, dtype=np.float32)
                - np.asarray(self.prev_reference_state.dof_pos, dtype=np.float32)
            ) / dt

        self.prev_reference_state = reference_state
        self.prev_reference_bodies = (body_pos, body_rot)
        self.reference_z_step_count += 1
        if self.reference_z_step_count <= self.drop_first_z_frames:
            return self.last_z.copy()

        ref_dof_pos = (np.asarray(reference_state.dof_pos, dtype=np.float32) - self.robot_config.default_joint_angles)[None, :]
        ref_dof_vel = dof_vel.astype(np.float32)[None, :]
        obs_dict = _compute_humanoid_observations_max_np(
            body_pos[None, :, :].astype(np.float32),
            body_rot[None, :, :].astype(np.float32),
            body_vel[None, :, :].astype(np.float32),
            body_ang_vel[None, :, :].astype(np.float32),
            local_root_obs=True,
            root_height_obs=self.robot_config.root_height_obs,
        )
        privileged_state = np.concatenate([v.astype(np.float32) for v in obs_dict.values()], axis=-1)
        projected_gravity = _quat_rotate_inverse_xyzw(
            body_rot[None, 0, :].astype(np.float32),
            np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
        )
        ref_ang_vel = body_ang_vel[None, 0, :].astype(np.float32) * self.robot_config.base_ang_vel_scale
        state = np.concatenate([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], axis=-1).astype(np.float32)
        last_action = np.zeros((1, self.robot_config.num_dof), dtype=np.float32)

        feed: dict[str, np.ndarray] = {}
        if "state" in self.backward_inputs:
            feed["state"] = state
        if "last_action" in self.backward_inputs:
            feed["last_action"] = last_action
        if "privileged_state" in self.backward_inputs:
            feed["privileged_state"] = privileged_state
        z = self._validate_z(self.backward_session.run([self.backward_output], feed)[0])
        if z is None:
            return self.last_z.copy()
        self.has_valid_reference_z = True
        return self._accept_z(z)

    def _smooth_z(self) -> np.ndarray:
        if not self.z_window:
            return self.last_z.copy()
        window = np.stack(list(self.z_window), axis=0)
        discounts = self.gamma ** np.arange(window.shape[0], dtype=np.float32)
        discounts = discounts / np.sum(discounts)
        z = np.sum(window * discounts[:, None], axis=0)
        norm = float(np.linalg.norm(z))
        if norm > 1e-8:
            z = z / norm * self.ctx_norm_ref
        return z.astype(np.float32)

    def _policy_segments(self, robot_state: RobotState, dt: float) -> dict[str, np.ndarray]:
        dof_pos = np.asarray(robot_state.dof_pos, dtype=np.float32).reshape(self.robot_config.num_dof)
        dof_vel, root_ang_vel = self._state_velocities(robot_state, self.prev_robot_state, dt)
        root_quat_xyzw = _wxyz_to_xyzw(np.asarray(robot_state.root_quat_wxyz, dtype=np.float32).reshape(4))
        projected_gravity = _quat_rotate_inverse_xyzw(
            root_quat_xyzw[None, :],
            np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
        ).reshape(3)
        base_ang_vel = root_ang_vel * self.robot_config.base_ang_vel_scale
        dof_pos_rel = dof_pos - self.robot_config.default_joint_angles
        state = np.concatenate([dof_pos_rel, dof_vel, projected_gravity, base_ang_vel], axis=0).astype(np.float32)

        history_dim = int(self.bundle.metadata.get("actor_input_dims", {}).get("history_actor", 0))
        if history_dim > 0:
            history_sources = {
                "actions": self.last_action.astype(np.float32),
                "base_ang_vel": base_ang_vel.astype(np.float32),
                "dof_pos": dof_pos_rel.astype(np.float32),
                "dof_vel": dof_vel.astype(np.float32),
                "projected_gravity": projected_gravity.astype(np.float32),
            }
            history_parts: list[np.ndarray] = []
            for key in sorted(self.history_buffers):
                rows = list(self.history_buffers[key])
                zero_row = np.zeros_like(history_sources[key], dtype=np.float32)
                while len(rows) < 4:
                    rows.append(zero_row)
                history_parts.append(np.concatenate(rows[:4], axis=0).astype(np.float32))
            history_actor = np.concatenate(history_parts, axis=0).astype(np.float32)
        else:
            history_actor = np.zeros(0, dtype=np.float32)

        return {
            "state": state,
            "last_action": self.last_action.astype(np.float32),
            "history_actor": history_actor,
            "_history_sources": {
                "actions": self.last_action.astype(np.float32),
                "base_ang_vel": base_ang_vel.astype(np.float32),
                "dof_pos": dof_pos_rel.astype(np.float32),
                "dof_vel": dof_vel.astype(np.float32),
                "projected_gravity": projected_gravity.astype(np.float32),
            },
        }

    def _policy_step_with_z(self, z: np.ndarray, robot_state: RobotState, dt: float) -> PolicyStepResult:
        z = np.asarray(z, dtype=np.float32).reshape(self.z_dim)
        z_valid = self._validate_z(z)
        if z_valid is None:
            z = self.last_z.copy()
        else:
            z = z_valid
            self.last_z = z.copy()
        now = time.monotonic()
        self.last_step_monotonic = now
        segments = self._policy_segments(robot_state, dt)
        actor_input_keys = [str(key) for key in self.bundle.metadata["actor_input_keys"]]
        actor_parts = [segments[key].reshape(1, -1) for key in actor_input_keys]
        actor_obs = np.concatenate([*actor_parts, z.reshape(1, -1)], axis=-1).astype(np.float32)

        expected = int(self.bundle.metadata["actor_obs_dim"])
        if actor_obs.shape[-1] != expected:
            raise ValueError(f"actor_obs dim mismatch: expected {expected}, got {actor_obs.shape[-1]}")

        raw_action = np.asarray(
            self.policy_session.run([self.policy_output], {self.policy_input: actor_obs})[0],
            dtype=np.float32,
        ).reshape(-1)
        if raw_action.shape != (self.robot_config.num_dof,):
            raise ValueError(f"policy action shape mismatch: expected {(self.robot_config.num_dof,)}, got {raw_action.shape}")
        if not np.all(np.isfinite(raw_action)):
            raise RuntimeError("policy produced non-finite action")

        normalized_action = np.clip(
            raw_action * self.robot_config.normalize_action_to,
            -self.robot_config.action_clip_value,
            self.robot_config.action_clip_value,
        ).astype(np.float32)
        q_target = self.robot_config.default_joint_angles + normalized_action * self.robot_config.action_target_scale
        q_target = np.clip(
            q_target,
            self.robot_config.joint_pos_lower_limit,
            self.robot_config.joint_pos_upper_limit,
        )
        baseline = self.last_q_target
        if baseline is None:
            baseline = np.asarray(robot_state.dof_pos, dtype=np.float32).reshape(self.robot_config.num_dof)
        max_delta = (
            self.robot_config.joint_velocity_limit
            * dt
            * self.robot_config.q_target_slew_safety_factor
        )
        if np.all(np.isfinite(max_delta)):
            q_target = baseline + np.clip(q_target - baseline, -max_delta, max_delta)
            q_target = np.clip(
                q_target,
                self.robot_config.joint_pos_lower_limit,
                self.robot_config.joint_pos_upper_limit,
            )

        self.last_action = normalized_action.copy()
        self.last_q_target = q_target.astype(np.float32).copy()
        for key, value in segments["_history_sources"].items():
            self.history_buffers[key].appendleft(np.asarray(value, dtype=np.float32).copy())
        self.prev_robot_state = robot_state
        return PolicyStepResult(
            z=z,
            actor_obs=actor_obs,
            raw_action=raw_action,
            normalized_action=normalized_action,
            q_target=q_target.astype(np.float32),
        )

    def current_z(self) -> np.ndarray:
        return self._smooth_z()

    def update_reference_z(self, reference_state: RobotState) -> np.ndarray:
        return self._reference_z(reference_state, self.dt)

    def reference_min_geom_z(self, reference_state: RobotState) -> float:
        return self.fk.min_geom_z_from_state(reference_state)

    def step_with_z(self, z: np.ndarray, robot_state: RobotState) -> PolicyStepResult:
        now = time.monotonic()
        dt = self.dt if self.last_step_monotonic is None else max(1.0 / 240.0, min(now - self.last_step_monotonic, 0.5))
        return self._policy_step_with_z(z, robot_state, dt)

    def step(
        self,
        reference_state: RobotState,
        robot_state: RobotState | None = None,
    ) -> PolicyStepResult:
        now = time.monotonic()
        dt = self.dt if self.last_step_monotonic is None else max(1.0 / 240.0, min(now - self.last_step_monotonic, 0.5))
        if robot_state is None:
            robot_state = reference_state

        z = self._reference_z(reference_state, dt)
        return self._policy_step_with_z(z, robot_state, dt)
