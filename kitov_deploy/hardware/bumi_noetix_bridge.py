"""Thin runtime bridge for Noetix BUMI low-level control."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kitov_deploy.policy_runtime import REPO_ROOT, RobotState


DEFAULT_BUMI_NOETIX_HARDWARE_CONFIG = REPO_ROOT / "configs" / "hardware" / "bumi_noetix.json"


@dataclass(frozen=True)
class BumiNoetixJointConfig:
    deploy_joint: str
    sdk_joint: str


@dataclass(frozen=True)
class BumiNoetixHardwareConfig:
    path: Path
    sdk_root: Path
    sdk_build_dir: Path
    dds_config: Path
    num_motors: int
    recv_timeout_s: float
    max_velocity_rad_s: float
    damping_kd: np.ndarray
    policy_kp: np.ndarray
    policy_kd: np.ndarray
    zero_kp: np.ndarray
    zero_kd: np.ndarray
    joints: list[BumiNoetixJointConfig]


@dataclass(frozen=True)
class BumiNoetixState:
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_quat_wxyz: np.ndarray
    root_ang_vel: np.ndarray
    timestamp: float


def _resolve_repo_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve(strict=False)


def _array(payload: dict[str, Any], name: str, length: int, path: Path) -> np.ndarray:
    values = np.asarray(payload[name], dtype=np.float32)
    if values.shape != (length,):
        raise ValueError(f"{path}: {name} must have shape ({length},), got {values.shape}")
    return values


def load_bumi_noetix_hardware_config(
    path: str | Path = DEFAULT_BUMI_NOETIX_HARDWARE_CONFIG,
) -> BumiNoetixHardwareConfig:
    config_path = _resolve_repo_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    joints = [
        BumiNoetixJointConfig(
            deploy_joint=str(item["deploy_joint"]),
            sdk_joint=str(item["sdk_joint"]),
        )
        for item in payload["joints"]
    ]
    num_motors = int(payload.get("num_motors", 21))
    dof_count = len(joints)
    return BumiNoetixHardwareConfig(
        path=config_path,
        sdk_root=_resolve_repo_path(payload.get("sdk_root", "third_party/noetix_sdk_bumi")),
        sdk_build_dir=_resolve_repo_path(payload.get("sdk_build_dir", "third_party/noetix_sdk_bumi/build")),
        dds_config=_resolve_repo_path(payload.get("dds_config", "third_party/noetix_sdk_bumi/config/dds.xml")),
        num_motors=num_motors,
        recv_timeout_s=float(payload.get("recv_timeout_s", 1.0)),
        max_velocity_rad_s=float(payload.get("max_velocity_rad_s", 1.0)),
        damping_kd=_array(payload, "damping_kd", dof_count, config_path),
        policy_kp=_array(payload, "policy_kp", dof_count, config_path),
        policy_kd=_array(payload, "policy_kd", dof_count, config_path),
        zero_kp=_array(payload, "zero_kp", dof_count, config_path),
        zero_kd=_array(payload, "zero_kd", dof_count, config_path),
        joints=joints,
    )


class BumiNoetixBridge:
    """Load ``lowcontrol_py`` at runtime and send Noetix BUMI motor commands."""

    def __init__(self, config: BumiNoetixHardwareConfig) -> None:
        self.config = config
        self.lowcontrol: Any | None = None
        self.ctrl: Any | None = None
        self._lock = threading.Lock()
        self._latest_status: Any | None = None
        self._latest_timestamp = 0.0
        self._hw_indices: list[int] = []

    def connect(self) -> None:
        if not self.config.dds_config.exists():
            raise FileNotFoundError(f"Noetix DDS config not found: {self.config.dds_config}")
        if not self.config.sdk_build_dir.exists():
            raise FileNotFoundError(
                "Noetix SDK build directory not found: "
                f"{self.config.sdk_build_dir}\n"
                "Build third_party/noetix_sdk_bumi first so lowcontrol_py is available."
            )

        os.environ["CYCLONEDDS_URI"] = "file://" + self.config.dds_config.as_posix()
        sys.path.insert(0, self.config.sdk_build_dir.as_posix())
        try:
            import lowcontrol_py  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import lowcontrol_py from Noetix SDK build directory:\n"
                f"  {self.config.sdk_build_dir}\n"
                "Run the Noetix SDK build first, then retry."
            ) from exc

        self.lowcontrol = lowcontrol_py
        self.ctrl = lowcontrol_py.LowController.instance()
        if not self.ctrl.init():
            raise RuntimeError("Noetix LowController.init() returned false")

        def _on_status(status: Any) -> None:
            with self._lock:
                self._latest_status = status
                self._latest_timestamp = time.monotonic()

        self.ctrl.subscribe_robot_hardware_status(_on_status)
        self._hw_indices = []
        for joint in self.config.joints:
            hw_idx = int(self.ctrl.getJointsIndex(joint.sdk_joint))
            if hw_idx < 0 or hw_idx >= self.config.num_motors:
                raise RuntimeError(f"Noetix SDK cannot resolve joint {joint.sdk_joint!r}: index={hw_idx}")
            self._hw_indices.append(hw_idx)

    def wait_for_state(self, timeout_s: float) -> BumiNoetixState | None:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while time.monotonic() <= deadline:
            state = self.read_state()
            if state is not None:
                return state
            time.sleep(0.01)
        return None

    def read_state(self) -> BumiNoetixState | None:
        with self._lock:
            status = self._latest_status
            timestamp = self._latest_timestamp
        if status is None:
            return None

        motor_data = status.motor_data
        dof_pos = np.zeros(len(self._hw_indices), dtype=np.float32)
        dof_vel = np.zeros(len(self._hw_indices), dtype=np.float32)
        for deploy_idx, hw_idx in enumerate(self._hw_indices):
            dof_pos[deploy_idx] = float(motor_data[hw_idx].pos)
            dof_vel[deploy_idx] = float(motor_data[hw_idx].vel)

        imu = status.imu_data
        quat_xyzw = np.asarray(imu.ori, dtype=np.float32).reshape(4)
        norm = max(float(np.linalg.norm(quat_xyzw)), 1e-8)
        quat_xyzw = quat_xyzw / norm
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]].astype(np.float32)
        root_ang_vel = np.asarray(imu.angular_vel, dtype=np.float32).reshape(3)
        return BumiNoetixState(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            root_quat_wxyz=quat_wxyz,
            root_ang_vel=root_ang_vel,
            timestamp=timestamp,
        )

    def robot_state(self, *, default_root_pos: np.ndarray) -> RobotState | None:
        state = self.read_state()
        if state is None:
            return None
        return RobotState(
            root_pos=np.asarray(default_root_pos, dtype=np.float32).reshape(3).copy(),
            root_quat_wxyz=state.root_quat_wxyz.copy(),
            dof_pos=state.dof_pos.copy(),
            root_lin_vel=np.zeros(3, dtype=np.float32),
            root_ang_vel=state.root_ang_vel.copy(),
            dof_vel=state.dof_vel.copy(),
        )

    def _empty_commands(self) -> list[Any]:
        if self.lowcontrol is None:
            raise RuntimeError("BumiNoetixBridge.connect() must be called first")
        commands = []
        for motor_id in range(self.config.num_motors):
            cmd = self.lowcontrol.MotorCmd()
            cmd.pos = 0.0
            cmd.vel = 0.0
            cmd.tau = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0
            cmd.motor_id = motor_id
            commands.append(cmd)
        return commands

    def send_damping(self) -> None:
        if self.ctrl is None:
            raise RuntimeError("BumiNoetixBridge.connect() must be called first")
        commands = self._empty_commands()
        for deploy_idx, hw_idx in enumerate(self._hw_indices):
            cmd = commands[hw_idx]
            cmd.pos = 0.0
            cmd.vel = 0.0
            cmd.tau = 0.0
            cmd.kp = 0.0
            cmd.kd = float(self.config.damping_kd[deploy_idx])
            cmd.motor_id = hw_idx
        self.ctrl.set_joint(commands)

    def send_positions(self, q_target: np.ndarray, *, kp: np.ndarray, kd: np.ndarray) -> None:
        if self.ctrl is None:
            raise RuntimeError("BumiNoetixBridge.connect() must be called first")
        q_target = np.asarray(q_target, dtype=np.float32).reshape(len(self._hw_indices))
        kp = np.asarray(kp, dtype=np.float32).reshape(len(self._hw_indices))
        kd = np.asarray(kd, dtype=np.float32).reshape(len(self._hw_indices))
        commands = self._empty_commands()
        for deploy_idx, hw_idx in enumerate(self._hw_indices):
            cmd = commands[hw_idx]
            cmd.pos = float(q_target[deploy_idx])
            cmd.vel = 0.0
            cmd.tau = 0.0
            cmd.kp = float(kp[deploy_idx])
            cmd.kd = float(kd[deploy_idx])
            cmd.motor_id = hw_idx
        self.ctrl.set_joint(commands)
