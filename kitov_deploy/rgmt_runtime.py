"""Runtime for BUMI RGMT policies."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kitov_deploy.mjcf_utils import prepared_mjcf_path
from kitov_deploy.policy_runtime import DEFAULT_MODEL_ROOT, REPO_ROOT, RobotPolicyConfig, RobotState, load_robot_policy_config


DEFAULT_RGMT_MODEL_DIR = DEFAULT_MODEL_ROOT / "bumi" / "rgmt"
DEFAULT_RGMT_CONFIG = REPO_ROOT / "configs" / "policy" / "bumi_rgmt.json"
GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)


@dataclass(frozen=True)
class BumiRGMTConfig:
    config_path: Path
    model_path: Path
    model_inputs: list[str]
    num_dofs: int
    policy_obs_dim: int
    history_length: int
    state_history_step_dim: int
    action_history_step_dim: int
    command_window_before: int
    command_window_after: int
    command_window_size: int
    command_step_dim: int
    anchor_body_name: str
    clip_obs: float
    action_mode: str
    joint_names: list[str]
    joint_mapping: np.ndarray
    default_dof_pos: np.ndarray
    action_scale: np.ndarray
    reference_joint_pos_lower: np.ndarray
    reference_joint_pos_upper: np.ndarray


@dataclass(frozen=True)
class RGMTReferenceFrame:
    qpos: np.ndarray
    dof_policy: np.ndarray
    anchor_quat_wxyz: np.ndarray
    anchor_lin_vel_world: np.ndarray
    anchor_ang_vel_world: np.ndarray


@dataclass(frozen=True)
class RGMTStepResult:
    policy_obs: np.ndarray
    state_history_obs: np.ndarray
    action_history_obs: np.ndarray
    command_obs: np.ndarray
    raw_action: np.ndarray
    q_target: np.ndarray


def _load_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import PyYAML. The default RGMT deploy config is JSON and "
            "does not need PyYAML; install PyYAML only if you pass a YAML config."
        ) from exc
    return yaml


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import torch, which is required by the BUMI RGMT TorchScript "
            "policy. Install a PyTorch build that matches this machine, then run "
            "this script again."
        ) from exc
    return torch


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Cannot import onnxruntime. Install dependencies with `uv sync --inexact`.") from exc
    return ort


def _as_array(payload: dict[str, Any], name: str, length: int, path: Path) -> np.ndarray:
    value = np.asarray(payload[name], dtype=np.float32)
    if value.shape != (length,):
        raise ValueError(f"{path}: {name} must have shape ({length},), got {value.shape}")
    return value


def _resolve_config_payload(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser()
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        yaml = _load_yaml()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: expected a config mapping")
    if "bumi/rgmt" in payload:
        payload = payload["bumi/rgmt"]
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: expected bumi/rgmt to be a config mapping")
    return payload


def load_bumi_rgmt_config(
    *,
    model_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    policy_path: Path | str | None = None,
) -> BumiRGMTConfig:
    model_dir_path = Path(model_dir).expanduser() if model_dir is not None else DEFAULT_RGMT_MODEL_DIR
    if config_path is not None:
        config_path_obj = Path(config_path).expanduser()
    else:
        model_config = model_dir_path / "config.json"
        config_path_obj = model_config if model_config.exists() else DEFAULT_RGMT_CONFIG
    payload = _resolve_config_payload(config_path_obj)

    if policy_path is not None:
        model_path = Path(policy_path).expanduser()
    else:
        model_name = str(payload.get("model_name", "policy.onnx"))
        candidates = [model_dir_path / "policy.onnx", model_dir_path / model_name, model_dir_path / "policy.pt"]
        seen: set[Path] = set()
        model_path = next((path for path in candidates if not (path in seen or seen.add(path)) and path.exists()), candidates[0])

    if not model_path.exists():
        raise FileNotFoundError(
            "Cannot find BUMI RGMT policy.\n"
            f"Expected: {model_path}\n"
            "Copy the exported policy to `models/bumi/rgmt/policy.onnx`, or pass "
            "`--policy-path /path/to/policy.onnx`. TorchScript `policy.pt` is still supported as a fallback."
        )

    num_dofs = int(payload.get("num_of_dofs", 21))
    joint_names = [str(name) for name in payload["joint_names"]]
    if len(joint_names) != num_dofs:
        raise ValueError(f"{config_path_obj}: joint_names length must be {num_dofs}")

    joint_mapping = np.asarray(payload["joint_mapping"], dtype=np.int32)
    if joint_mapping.shape != (num_dofs,):
        raise ValueError(f"{config_path_obj}: joint_mapping must have shape ({num_dofs},)")
    if sorted(int(v) for v in joint_mapping) != list(range(num_dofs)):
        raise ValueError(f"{config_path_obj}: joint_mapping must be a permutation of 0..{num_dofs - 1}")

    return BumiRGMTConfig(
        config_path=config_path_obj,
        model_path=model_path,
        model_inputs=[str(name) for name in payload.get("model_inputs", [])],
        num_dofs=num_dofs,
        policy_obs_dim=int(payload.get("rgmt_policy_obs_dim", 69)),
        history_length=int(payload.get("rgmt_history_length", 10)),
        state_history_step_dim=int(payload.get("rgmt_state_history_step_dim", 48)),
        action_history_step_dim=int(payload.get("rgmt_action_history_step_dim", 21)),
        command_window_before=int(payload.get("rgmt_command_window_before", 10)),
        command_window_after=int(payload.get("rgmt_command_window_after", 10)),
        command_window_size=int(payload.get("rgmt_command_window_size", 21)),
        command_step_dim=int(payload.get("rgmt_command_step_dim", 30)),
        anchor_body_name=str(payload.get("rgmt_anchor_body_name", "waist_yaw_link")),
        clip_obs=float(payload.get("clip_obs", 100.0)),
        action_mode=str(payload.get("action_mode", "reference_joint_position")),
        joint_names=joint_names,
        joint_mapping=joint_mapping,
        default_dof_pos=_as_array(payload, "default_dof_pos", num_dofs, config_path_obj),
        action_scale=_as_array(payload, "action_scale", num_dofs, config_path_obj),
        reference_joint_pos_lower=_as_array(payload, "reference_joint_pos_lower", num_dofs, config_path_obj),
        reference_joint_pos_upper=_as_array(payload, "reference_joint_pos_upper", num_dofs, config_path_obj),
    )


def _qpos_from_state(state: RobotState) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state.root_pos, dtype=np.float64).reshape(3),
            np.asarray(state.root_quat_wxyz, dtype=np.float64).reshape(4),
            np.asarray(state.dof_pos, dtype=np.float64).reshape(-1),
        ],
        axis=0,
    )


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return q[..., [1, 2, 3, 0]]


def _quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_w = q[..., 3:4]
    q_vec = q[..., 0:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def _quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyzw = _wxyz_to_xyzw(np.asarray(q, dtype=np.float32))
    norm = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    q_xyzw = q_xyzw / np.clip(norm, 1e-8, None)
    return _quat_rotate_inverse_xyzw(q_xyzw, np.asarray(v, dtype=np.float32))


class BumiRGMTRuntime:
    """Run BUMI RGMT policy from live GMR references and MuJoCo robot state."""

    def __init__(
        self,
        *,
        model_dir: Path | str | None = None,
        config_path: Path | str | None = None,
        policy_path: Path | str | None = None,
        hz: float = 50.0,
        device: str = "cpu",
    ) -> None:
        self.config = load_bumi_rgmt_config(
            model_dir=model_dir,
            config_path=config_path,
            policy_path=policy_path,
        )
        if self.config.action_mode != "reference_joint_position":
            raise ValueError(f"Unsupported RGMT action_mode: {self.config.action_mode!r}")

        self.robot_config = load_robot_policy_config("bumi")
        if self.robot_config.num_dof != self.config.num_dofs:
            raise ValueError(
                f"BUMI robot config has {self.robot_config.num_dof} DoF, "
                f"RGMT config has {self.config.num_dofs} DoF"
            )

        self._validate_joint_mapping()
        self._policy_to_deploy = self.config.joint_mapping.astype(np.int32)
        self._deploy_to_policy = np.empty_like(self._policy_to_deploy)
        self._deploy_to_policy[self._policy_to_deploy] = np.arange(self.config.num_dofs, dtype=np.int32)
        self.dt = 1.0 / max(float(hz), 1e-6)

        self.backend = self.config.model_path.suffix.lower()
        self.device_name = str(device)
        self.torch = None
        self.ort = None
        self.model = None
        self._onnx_input_names: list[str] = []
        if self.backend == ".onnx":
            self.ort = _load_onnxruntime()
            if self.device_name.startswith("cuda"):
                available = set(self.ort.get_available_providers())
                if "CUDAExecutionProvider" not in available:
                    raise RuntimeError(
                        "`--device cuda` was requested for RGMT ONNX, but onnxruntime "
                        "does not expose CUDAExecutionProvider. Install onnxruntime-gpu "
                        "or run without `--device cuda`."
                    )
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self.model = self.ort.InferenceSession(str(self.config.model_path), providers=providers)
            self._onnx_input_names = [inp.name for inp in self.model.get_inputs()]
        elif self.backend == ".pt":
            self.torch = _load_torch()
            self.device = self.torch.device(device)
            if self.device.type == "cuda" and not self.torch.cuda.is_available():
                raise RuntimeError("`--device cuda` was requested, but torch.cuda.is_available() is false.")
            self.model = self.torch.jit.load(str(self.config.model_path), map_location=self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported RGMT policy file extension: {self.config.model_path}")

        self.mujoco = self._load_mujoco()
        if not self.robot_config.xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {self.robot_config.xml_path}")
        self._prepared_context = prepared_mjcf_path(self.robot_config.xml_path)
        prepared_xml_path = self._prepared_context.__enter__()
        try:
            self.fk_model = self.mujoco.MjModel.from_xml_path(str(prepared_xml_path))
        except Exception:
            self._prepared_context.__exit__(None, None, None)
            raise
        self.fk_data = self.mujoco.MjData(self.fk_model)
        if self.robot_config.sim_timestep is not None:
            self.fk_model.opt.timestep = float(self.robot_config.sim_timestep)
        try:
            self.anchor_body_id = int(self.fk_model.body(self.config.anchor_body_name).id)
        except KeyError as exc:
            self.close()
            raise KeyError(
                f"RGMT anchor body {self.config.anchor_body_name!r} not found in "
                f"{self.robot_config.xml_path}"
            ) from exc

        self._reference_frames: deque[RGMTReferenceFrame] = deque(maxlen=self.config.command_window_size)
        self._state_history: deque[np.ndarray] = deque(maxlen=self.config.history_length)
        self._action_history: deque[np.ndarray] = deque(maxlen=self.config.history_length)
        self._last_ref_qpos: np.ndarray | None = None
        self._last_action = np.zeros(self.config.num_dofs, dtype=np.float32)

    def _validate_joint_mapping(self) -> None:
        for policy_index, deploy_index in enumerate(self.config.joint_mapping):
            expected = self.config.joint_names[policy_index]
            actual = self.robot_config.control_joint_names[int(deploy_index)]
            if expected != actual:
                raise ValueError(
                    "BUMI RGMT joint_mapping does not match deploy config: "
                    f"policy[{policy_index}]={expected!r} maps to deploy[{int(deploy_index)}]={actual!r}"
                )

    @staticmethod
    def _load_mujoco() -> Any:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Cannot import mujoco. Install dependencies with `uv sync --inexact`.") from exc
        return mujoco

    def close(self) -> None:
        context = getattr(self, "_prepared_context", None)
        if context is not None:
            context.__exit__(None, None, None)
            self._prepared_context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def deploy_to_policy(self, dof_pos: np.ndarray) -> np.ndarray:
        dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(self.config.num_dofs)
        return dof_pos[self._policy_to_deploy].astype(np.float32).copy()

    def policy_to_deploy(self, dof_pos: np.ndarray) -> np.ndarray:
        dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(self.config.num_dofs)
        out = np.empty(self.config.num_dofs, dtype=np.float32)
        out[self._policy_to_deploy] = dof_pos
        return out

    def reference_min_geom_z(self, reference_state: RobotState) -> float:
        self.fk_data.qpos[:] = _qpos_from_state(reference_state)
        self.fk_data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.fk_model, self.fk_data)
        return float(np.min(self.fk_data.geom_xpos[:, 2]))

    def update_reference(self, reference_state: RobotState) -> RGMTReferenceFrame:
        qpos = _qpos_from_state(reference_state)
        qvel = np.zeros(self.fk_model.nv, dtype=np.float64)
        if self._last_ref_qpos is not None:
            self.mujoco.mj_differentiatePos(self.fk_model, qvel, self.dt, self._last_ref_qpos, qpos)

        self.fk_data.qpos[:] = qpos
        self.fk_data.qvel[:] = qvel
        self.mujoco.mj_forward(self.fk_model, self.fk_data)
        self.mujoco.mj_fwdVelocity(self.fk_model, self.fk_data)

        anchor_vel = np.zeros(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.fk_model,
            self.fk_data,
            self.mujoco.mjtObj.mjOBJ_BODY,
            self.anchor_body_id,
            anchor_vel,
            0,
        )
        frame = RGMTReferenceFrame(
            qpos=qpos.astype(np.float32).copy(),
            dof_policy=self.deploy_to_policy(reference_state.dof_pos),
            anchor_quat_wxyz=self.fk_data.xquat[self.anchor_body_id].astype(np.float32).copy(),
            anchor_lin_vel_world=anchor_vel[3:6].astype(np.float32).copy(),
            anchor_ang_vel_world=anchor_vel[:3].astype(np.float32).copy(),
        )
        self._reference_frames.append(frame)
        self._last_ref_qpos = qpos.copy()
        return frame

    def _state_obs(self, robot_state: RobotState) -> np.ndarray:
        root_quat = np.asarray(robot_state.root_quat_wxyz, dtype=np.float32).reshape(4)
        projected_gravity = _quat_rotate_inverse_wxyz(root_quat, GRAVITY_VEC)
        root_ang_vel = (
            np.zeros(3, dtype=np.float32)
            if robot_state.root_ang_vel is None
            else np.asarray(robot_state.root_ang_vel, dtype=np.float32).reshape(3)
        )
        dof_pos = self.deploy_to_policy(robot_state.dof_pos)
        if robot_state.dof_vel is None:
            dof_vel = np.zeros(self.config.num_dofs, dtype=np.float32)
        else:
            dof_vel = self.deploy_to_policy(robot_state.dof_vel)
        return np.concatenate(
            [
                projected_gravity,
                root_ang_vel,
                dof_pos - self.config.default_dof_pos,
                dof_vel,
            ],
            axis=0,
        ).astype(np.float32)

    def _ensure_history(self, state_obs: np.ndarray) -> None:
        if self._state_history:
            self._state_history.append(state_obs.astype(np.float32).copy())
            self._action_history.append(self._last_action.astype(np.float32).copy())
            return
        for _ in range(self.config.history_length):
            self._state_history.append(state_obs.astype(np.float32).copy())
            self._action_history.append(self._last_action.astype(np.float32).copy())

    @property
    def reference_buffer_size(self) -> int:
        return len(self._reference_frames)

    def can_step_from_buffer(self, *, delay_frames: int) -> bool:
        delay = max(int(delay_frames), 0)
        if len(self._reference_frames) <= delay:
            return False
        center_index = len(self._reference_frames) - 1 - delay
        return center_index >= self.config.command_window_before

    def _reference_at_offset(self, offset: int, *, center_index: int | None = None) -> RGMTReferenceFrame:
        if not self._reference_frames:
            raise RuntimeError("RGMT reference command requested before any reference frame was added.")
        references = list(self._reference_frames)
        if center_index is None:
            if offset >= 0:
                return references[-1]
            index = max(-len(references), offset - 1)
            return references[index]

        index = min(max(int(center_index) + int(offset), 0), len(references) - 1)
        return references[index]

    def _command_window(self, *, center_index: int | None = None) -> np.ndarray:
        steps: list[np.ndarray] = []
        for offset in range(-self.config.command_window_before, self.config.command_window_after + 1):
            frame = self._reference_at_offset(offset, center_index=center_index)
            lin_vel_b = _quat_rotate_inverse_wxyz(frame.anchor_quat_wxyz, frame.anchor_lin_vel_world)
            ang_vel_b = _quat_rotate_inverse_wxyz(frame.anchor_quat_wxyz, frame.anchor_ang_vel_world)
            gravity_b = _quat_rotate_inverse_wxyz(frame.anchor_quat_wxyz, GRAVITY_VEC)
            steps.append(
                np.concatenate([lin_vel_b, ang_vel_b, gravity_b, frame.dof_policy], axis=0).astype(np.float32)
            )
        command = np.stack(steps, axis=0)
        if command.shape != (self.config.command_window_size, self.config.command_step_dim):
            raise RuntimeError(
                "RGMT command shape mismatch: "
                f"expected {(self.config.command_window_size, self.config.command_step_dim)}, got {command.shape}"
            )
        return command

    def _step_with_reference(
        self,
        *,
        reference_frame: RGMTReferenceFrame,
        robot_state: RobotState,
        command_obs: np.ndarray,
    ) -> RGMTStepResult:
        state_obs = self._state_obs(robot_state)
        self._ensure_history(state_obs)

        policy_obs = np.concatenate([state_obs, self._last_action], axis=0).astype(np.float32)
        state_history_obs = np.stack(list(self._state_history), axis=0).astype(np.float32)
        action_history_obs = np.stack(list(self._action_history), axis=0).astype(np.float32)

        if policy_obs.shape != (self.config.policy_obs_dim,):
            raise RuntimeError(f"RGMT policy_obs shape mismatch: got {policy_obs.shape}")
        if state_history_obs.shape != (self.config.history_length, self.config.state_history_step_dim):
            raise RuntimeError(f"RGMT state_history_obs shape mismatch: got {state_history_obs.shape}")
        if action_history_obs.shape != (self.config.history_length, self.config.action_history_step_dim):
            raise RuntimeError(f"RGMT action_history_obs shape mismatch: got {action_history_obs.shape}")

        clip = float(self.config.clip_obs)
        policy_obs = np.clip(policy_obs, -clip, clip)
        state_history_obs = np.clip(state_history_obs, -clip, clip)
        action_history_obs = np.clip(action_history_obs, -clip, clip)
        command_obs = np.clip(command_obs.astype(np.float32), -clip, clip)

        raw_action = self._forward_action(policy_obs, state_history_obs, action_history_obs, command_obs)
        if raw_action.shape != (self.config.num_dofs,):
            raise RuntimeError(f"RGMT action shape mismatch: expected ({self.config.num_dofs},), got {raw_action.shape}")

        q_target_policy = reference_frame.dof_policy + raw_action * self.config.action_scale
        q_target_policy = np.clip(
            q_target_policy,
            self.config.reference_joint_pos_lower,
            self.config.reference_joint_pos_upper,
        ).astype(np.float32)
        q_target = self.policy_to_deploy(q_target_policy)
        self._last_action = raw_action.copy()

        return RGMTStepResult(
            policy_obs=policy_obs,
            state_history_obs=state_history_obs,
            action_history_obs=action_history_obs,
            command_obs=command_obs,
            raw_action=raw_action,
            q_target=q_target,
        )

    def step_from_buffer(self, *, delay_frames: int, robot_state: RobotState) -> RGMTStepResult:
        delay = max(int(delay_frames), 0)
        if not self.can_step_from_buffer(delay_frames=delay):
            raise RuntimeError(
                "Not enough RGMT reference frames for delayed policy step: "
                f"buffer={len(self._reference_frames)}, delay_frames={delay}"
            )
        center_index = len(self._reference_frames) - 1 - delay
        reference_frame = list(self._reference_frames)[center_index]
        command_obs = self._command_window(center_index=center_index).astype(np.float32)
        return self._step_with_reference(
            reference_frame=reference_frame,
            robot_state=robot_state,
            command_obs=command_obs,
        )

    def reference_state_from_buffer(self, *, delay_frames: int) -> RobotState:
        delay = max(int(delay_frames), 0)
        if not self.can_step_from_buffer(delay_frames=delay):
            raise RuntimeError(
                "Not enough RGMT reference frames for delayed reference state: "
                f"buffer={len(self._reference_frames)}, delay_frames={delay}"
            )
        reference_frame = list(self._reference_frames)[len(self._reference_frames) - 1 - delay]
        qpos = reference_frame.qpos
        return RobotState(
            root_pos=np.asarray(qpos[:3], dtype=np.float32).copy(),
            root_quat_wxyz=np.asarray(qpos[3:7], dtype=np.float32).copy(),
            dof_pos=np.asarray(qpos[7:], dtype=np.float32).copy(),
        )

    def _forward_action(
        self,
        policy_obs: np.ndarray,
        state_history_obs: np.ndarray,
        action_history_obs: np.ndarray,
        command_obs: np.ndarray,
    ) -> np.ndarray:
        if self.backend == ".onnx":
            if self.model is None:
                raise RuntimeError("RGMT ONNX session is not initialized.")
            tensors = {
                "rgmt_policy": policy_obs[None, :].astype(np.float32),
                "rgmt_state_history": state_history_obs[None, :, :].astype(np.float32),
                "rgmt_action_history": action_history_obs[None, :, :].astype(np.float32),
                "rgmt_command": command_obs[None, :, :].astype(np.float32),
            }
            if all(name in tensors for name in self._onnx_input_names):
                feed = {name: tensors[name] for name in self._onnx_input_names}
            elif len(self._onnx_input_names) == 4:
                ordered = [
                    tensors["rgmt_policy"],
                    tensors["rgmt_state_history"],
                    tensors["rgmt_action_history"],
                    tensors["rgmt_command"],
                ]
                feed = dict(zip(self._onnx_input_names, ordered, strict=True))
            else:
                raise RuntimeError(
                    "Cannot map RGMT ONNX inputs. Expected names "
                    "rgmt_policy, rgmt_state_history, rgmt_action_history, rgmt_command "
                    f"or exactly 4 positional inputs, got {self._onnx_input_names}"
                )
            return self.model.run(None, feed)[0].reshape(-1).astype(np.float32)

        if self.backend == ".pt":
            if self.model is None or self.torch is None:
                raise RuntimeError("RGMT TorchScript model is not initialized.")
            torch = self.torch
            with torch.no_grad():
                action_tensor = self.model(
                    torch.as_tensor(policy_obs[None, :], dtype=torch.float32, device=self.device),
                    torch.as_tensor(state_history_obs[None, :, :], dtype=torch.float32, device=self.device),
                    torch.as_tensor(action_history_obs[None, :, :], dtype=torch.float32, device=self.device),
                    torch.as_tensor(command_obs[None, :, :], dtype=torch.float32, device=self.device),
                )
            return action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)

        raise RuntimeError(f"Unsupported RGMT backend: {self.backend}")

    def step(self, *, reference_state: RobotState, robot_state: RobotState) -> RGMTStepResult:
        reference_frame = self.update_reference(reference_state)
        command_obs = self._command_window().astype(np.float32)
        return self._step_with_reference(
            reference_frame=reference_frame,
            robot_state=robot_state,
            command_obs=command_obs,
        )
