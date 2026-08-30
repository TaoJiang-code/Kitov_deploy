"""OpenArm v1 CAN hardware bridge.

The module is intentionally conservative: importing it never imports
``openarm_can`` and never touches hardware. The SDK is loaded only by
``OpenArmCANBridge.connect()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ee_body import load_ee_body
from kitov_deploy.mjcf_utils import prepared_mjcf_path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENARM_HARDWARE_CONFIG = REPO_ROOT / "configs" / "hardware" / "openarm_v1.json"


@dataclass(frozen=True)
class OpenArmMotorConfig:
    side: str
    joint_name: str
    motor_type: str
    send_can_id: int
    recv_can_id: int
    sign: float
    zero_offset: float
    kp: float
    kd: float
    max_velocity_rad_s: float
    hardware_lower: float | None
    hardware_upper: float | None
    enabled: bool = True

    @property
    def key(self) -> str:
        return self.joint_name

    def sim_to_hardware(self, sim_q: float) -> float:
        return float(self.sign) * float(sim_q) + float(self.zero_offset)

    def hardware_to_sim(self, hardware_q: float) -> float:
        if abs(float(self.sign)) < 1e-8:
            raise ValueError(f"Motor {self.joint_name} has invalid sign={self.sign}")
        return (float(hardware_q) - float(self.zero_offset)) / float(self.sign)


@dataclass(frozen=True)
class OpenArmBusConfig:
    side: str
    interface: str
    can_fd: bool
    motors: tuple[OpenArmMotorConfig, ...]
    enabled: bool = True


@dataclass(frozen=True)
class OpenArmSafetyConfig:
    recv_timeout_us: int
    enable_recv_timeout_us: int
    default_max_velocity_rad_s: float


@dataclass(frozen=True)
class OpenArmHardwareConfig:
    robot: str
    xml_path: Path
    buses: tuple[OpenArmBusConfig, ...]
    safety: OpenArmSafetyConfig
    ee_body_name: str | None = None
    ee_body: Any | None = None

    @property
    def motors(self) -> tuple[OpenArmMotorConfig, ...]:
        return tuple(motor for bus in self.buses if bus.enabled for motor in bus.motors if motor.enabled)


@dataclass(frozen=True)
class MotorState:
    joint_name: str
    position: float
    velocity: float | None
    torque: float | None
    enabled: bool | None


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve(strict=False)


def _parse_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def load_openarm_hardware_config(path: str | Path = DEFAULT_OPENARM_HARDWARE_CONFIG) -> OpenArmHardwareConfig:
    config_path = _resolve_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    safety_payload = dict(payload.get("safety", {}))
    safety = OpenArmSafetyConfig(
        recv_timeout_us=int(safety_payload.get("recv_timeout_us", 500)),
        enable_recv_timeout_us=int(safety_payload.get("enable_recv_timeout_us", 500000)),
        default_max_velocity_rad_s=float(safety_payload.get("default_max_velocity_rad_s", 0.5)),
    )

    buses: list[OpenArmBusConfig] = []
    for bus_payload in payload["buses"]:
        side = str(bus_payload["side"])
        motors = []
        for motor_payload in bus_payload.get("motors", []):
            motors.append(
                OpenArmMotorConfig(
                    side=side,
                    joint_name=str(motor_payload["joint_name"]),
                    motor_type=str(motor_payload["motor_type"]),
                    send_can_id=_parse_int(motor_payload["send_can_id"]),
                    recv_can_id=_parse_int(motor_payload["recv_can_id"]),
                    sign=float(motor_payload.get("sign", 1.0)),
                    zero_offset=float(motor_payload.get("zero_offset", 0.0)),
                    kp=float(motor_payload.get("kp", payload.get("default_kp", 1.0))),
                    kd=float(motor_payload.get("kd", payload.get("default_kd", 0.2))),
                    max_velocity_rad_s=float(
                        motor_payload.get("max_velocity_rad_s", safety.default_max_velocity_rad_s)
                    ),
                    hardware_lower=(
                        None
                        if motor_payload.get("hardware_lower") is None
                        else float(motor_payload["hardware_lower"])
                    ),
                    hardware_upper=(
                        None
                        if motor_payload.get("hardware_upper") is None
                        else float(motor_payload["hardware_upper"])
                    ),
                    enabled=bool(motor_payload.get("enabled", True)),
                )
            )
        buses.append(
            OpenArmBusConfig(
                side=side,
                interface=str(bus_payload["interface"]),
                can_fd=bool(bus_payload.get("can_fd", True)),
                motors=tuple(motors),
                enabled=bool(bus_payload.get("enabled", True)),
            )
        )

    ee_body_name = payload.get("ee_body", payload.get("ee_body_config"))
    ee_body = load_ee_body(ee_body_name)

    return OpenArmHardwareConfig(
        robot=str(payload.get("robot", "openarm_v1")),
        xml_path=_resolve_path(payload["xml_path"]),
        buses=tuple(buses),
        safety=safety,
        ee_body_name=None if ee_body_name is None else str(ee_body_name),
        ee_body=ee_body,
    )


class OpenArmQposMapper:
    """Map OpenArm MuJoCo qpos vectors to hardware motor targets."""

    def __init__(self, config: OpenArmHardwareConfig) -> None:
        self.config = config
        self.mujoco = self._load_mujoco()
        with prepared_mjcf_path(config.xml_path) as xml_path:
            self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self._joint_qpos_addr: dict[str, int] = {}
        self._joint_ranges: dict[str, tuple[float, float]] = {}
        for motor in config.motors:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, motor.joint_name)
            if joint_id < 0:
                raise KeyError(f"Hardware config references missing MuJoCo joint: {motor.joint_name}")
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            self._joint_qpos_addr[motor.joint_name] = qpos_addr
            if bool(self.model.jnt_limited[joint_id]):
                lower, upper = self.model.jnt_range[joint_id]
                self._joint_ranges[motor.joint_name] = (float(lower), float(upper))

    @staticmethod
    def _load_mujoco() -> Any:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import mujoco. Activate Kitov_deploy and install dependencies first:\n"
                "python -m pip install mujoco"
            ) from exc
        return mujoco

    def sim_joint_positions(self, qpos: np.ndarray) -> dict[str, float]:
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos_arr.shape[0] != self.model.nq:
            raise ValueError(f"OpenArm qpos size mismatch: got {qpos_arr.shape[0]}, expected {self.model.nq}")
        values = {}
        for joint_name, qpos_addr in self._joint_qpos_addr.items():
            sim_q = float(qpos_arr[qpos_addr])
            if joint_name in self._joint_ranges:
                lower, upper = self._joint_ranges[joint_name]
                sim_q = float(np.clip(sim_q, lower, upper))
            values[joint_name] = sim_q
        return values

    def sim_joint_limits(self) -> dict[str, tuple[float | None, float | None]]:
        return {motor.joint_name: self._joint_ranges.get(motor.joint_name, (None, None)) for motor in self.config.motors}

    def qpos_from_sim_joint_positions(self, reference_qpos: np.ndarray, positions: dict[str, float]) -> np.ndarray:
        qpos = np.asarray(reference_qpos, dtype=np.float64).reshape(-1).copy()
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"OpenArm qpos size mismatch: got {qpos.shape[0]}, expected {self.model.nq}")

        for joint_name, sim_q_raw in positions.items():
            if joint_name not in self._joint_qpos_addr:
                continue
            sim_q = float(sim_q_raw)
            if joint_name in self._joint_ranges:
                lower, upper = self._joint_ranges[joint_name]
                sim_q = float(np.clip(sim_q, lower, upper))
            qpos[self._joint_qpos_addr[joint_name]] = sim_q
        return qpos

    def hardware_targets_from_qpos(self, qpos: np.ndarray) -> dict[str, float]:
        sim_q_by_joint = self.sim_joint_positions(qpos)
        targets = {}
        for motor in self.config.motors:
            target = motor.sim_to_hardware(sim_q_by_joint[motor.joint_name])
            lower, upper = self.hardware_limit(motor)
            if lower is not None:
                target = max(target, lower)
            if upper is not None:
                target = min(target, upper)
            targets[motor.joint_name] = float(target)
        return targets

    def hardware_limit(self, motor: OpenArmMotorConfig) -> tuple[float | None, float | None]:
        lower = motor.hardware_lower
        upper = motor.hardware_upper
        if lower is None or upper is None:
            xml_limit = self._joint_ranges.get(motor.joint_name)
            if xml_limit is not None:
                xml_lower, xml_upper = xml_limit
                hardware_xml_limits = sorted(
                    (motor.sim_to_hardware(xml_lower), motor.sim_to_hardware(xml_upper))
                )
                if lower is None:
                    lower = float(hardware_xml_limits[0])
                if upper is None:
                    upper = float(hardware_xml_limits[1])
        return lower, upper

    def hardware_limits(self) -> dict[str, tuple[float | None, float | None]]:
        return {motor.joint_name: self.hardware_limit(motor) for motor in self.config.motors}

    def qpos_from_hardware_targets(self, reference_qpos: np.ndarray, targets: dict[str, float]) -> np.ndarray:
        qpos = np.asarray(reference_qpos, dtype=np.float64).reshape(-1).copy()
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"OpenArm qpos size mismatch: got {qpos.shape[0]}, expected {self.model.nq}")

        motor_by_joint = {motor.joint_name: motor for motor in self.config.motors}
        for joint_name, hardware_q in targets.items():
            if joint_name not in motor_by_joint:
                continue
            sim_q = motor_by_joint[joint_name].hardware_to_sim(float(hardware_q))
            if joint_name in self._joint_ranges:
                lower, upper = self._joint_ranges[joint_name]
                sim_q = float(np.clip(sim_q, lower, upper))
            qpos[self._joint_qpos_addr[joint_name]] = sim_q
        return qpos


class OpenArmCommandLimiter:
    """Rate-limit hardware targets to avoid jumps when retargeting starts."""

    def __init__(self, config: OpenArmHardwareConfig) -> None:
        self.config = config
        self._previous: dict[str, float] | None = None

    def reset(self, targets: dict[str, float] | None = None) -> None:
        self._previous = None if targets is None else {str(k): float(v) for k, v in targets.items()}

    def limit(self, targets: dict[str, float], dt: float) -> dict[str, float]:
        clean_targets = {str(k): float(v) for k, v in targets.items()}
        if self._previous is None:
            self._previous = dict(clean_targets)
            return clean_targets

        dt = max(float(dt), 1e-6)
        motor_by_joint = {motor.joint_name: motor for motor in self.config.motors}
        limited: dict[str, float] = {}
        for joint_name, target in clean_targets.items():
            prev = float(self._previous.get(joint_name, target))
            max_step = max(float(motor_by_joint[joint_name].max_velocity_rad_s) * dt, 0.0)
            limited[joint_name] = float(prev + np.clip(target - prev, -max_step, max_step))
        self._previous = dict(limited)
        return limited


class OpenArmCANBridge:
    """Thin wrapper around the experimental ``openarm_can`` Python binding."""

    def __init__(self, config: OpenArmHardwareConfig) -> None:
        self.config = config
        self.oa: Any | None = None
        self._arms: dict[str, Any] = {}
        self._active_buses: tuple[OpenArmBusConfig, ...] = tuple(bus for bus in config.buses if bus.enabled)
        self._ee_body = config.ee_body
        self.connected = False

    def connect(self) -> None:
        self.oa = self._load_openarm_can()
        for bus in self._active_buses:
            arm = self.oa.OpenArm(bus.interface, bus.can_fd)
            motors = [motor for motor in bus.motors if motor.enabled]
            motor_types = [self._motor_type(motor.motor_type) for motor in motors]
            send_ids = [motor.send_can_id for motor in motors]
            recv_ids = [motor.recv_can_id for motor in motors]
            control_modes = [self.oa.ControlMode.MIT for _ in motors]
            arm.init_arm_motors(motor_types, send_ids, recv_ids, control_modes)
            if self._ee_body is not None:
                self._ee_body.init_bus(
                    side=bus.side,
                    arm=arm,
                    motor_type=self._motor_type,
                    control_mode=self._control_mode,
                )
            arm.set_callback_mode_all(self.oa.CallbackMode.STATE)
            self._arms[bus.side] = arm
        self.connected = True

    @staticmethod
    def _load_openarm_can() -> Any:
        try:
            import openarm_can as oa
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import openarm_can. Install the OpenArm CAN C++ library and Python binding first.\n"
                "Suggested source path in this repo: third_party/openarm_can"
            ) from exc
        return oa

    def _motor_type(self, name: str) -> Any:
        assert self.oa is not None
        if not hasattr(self.oa.MotorType, name):
            raise RuntimeError(f"openarm_can.MotorType has no member {name!r}")
        return getattr(self.oa.MotorType, name)

    def _control_mode(self, name: str) -> Any:
        assert self.oa is not None
        mode_name = str(name).upper()
        if not hasattr(self.oa.ControlMode, mode_name):
            raise RuntimeError(f"openarm_can.ControlMode has no member {mode_name!r}")
        return getattr(self.oa.ControlMode, mode_name)

    def enable_all(self) -> None:
        self._require_connected()
        for arm in self._arms.values():
            arm.set_callback_mode_all(self.oa.CallbackMode.IGNORE)
            arm.enable_all()
            arm.recv_all(self.config.safety.enable_recv_timeout_us)
            arm.set_callback_mode_all(self.oa.CallbackMode.STATE)

    def disable_all(self) -> None:
        if not self.connected:
            return
        for arm in self._arms.values():
            arm.disable_all()
            arm.recv_all(self.config.safety.enable_recv_timeout_us)

    def set_zero_all(self) -> None:
        self._require_connected()
        for arm in self._arms.values():
            arm.set_callback_mode_all(self.oa.CallbackMode.IGNORE)
            arm.set_zero_all()
            arm.recv_all(self.config.safety.enable_recv_timeout_us)
            arm.set_callback_mode_all(self.oa.CallbackMode.STATE)

    def read_state(self, *, recv_timeout_us: int | None = None) -> dict[str, MotorState]:
        self._require_connected()
        timeout_us = self.config.safety.recv_timeout_us if recv_timeout_us is None else int(recv_timeout_us)
        states: dict[str, MotorState] = {}
        for bus in self._active_buses:
            arm = self._arms[bus.side]
            arm.refresh_all()
            arm.recv_all(timeout_us)
            motors = [motor for motor in bus.motors if motor.enabled]
            for motor_config, motor in zip(motors, arm.get_arm().get_motors(), strict=False):
                states[motor_config.joint_name] = MotorState(
                    joint_name=motor_config.joint_name,
                    position=float(motor.get_position()),
                    velocity=_optional_float_call(motor, "get_velocity"),
                    torque=_optional_float_call(motor, "get_torque"),
                    enabled=_optional_bool_call(motor, "is_enabled"),
                )
            if self._ee_body is not None:
                states.update(
                    self._ee_body.read_state(
                        side=bus.side,
                        arm=arm,
                        make_state=MotorState,
                        optional_float=_optional_float_call,
                        optional_bool=_optional_bool_call,
                    )
                )
        return states

    def send_position_targets(
        self,
        targets: dict[str, float],
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        recv: bool = True,
    ) -> None:
        self._require_connected()
        assert self.oa is not None
        for bus in self._active_buses:
            arm = self._arms[bus.side]
            params = []
            for motor in bus.motors:
                if not motor.enabled:
                    continue
                params.append(
                    self.oa.MITParam(
                        float(motor.kp) * float(kp_scale),
                        float(motor.kd) * float(kd_scale),
                        float(targets[motor.joint_name]),
                        0.0,
                        0.0,
                    )
                )
            arm.get_arm().mit_control_all(params)
            if recv:
                arm.recv_all(self.config.safety.recv_timeout_us)

    def send_ee_positions(
        self,
        positions: dict[str, float],
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        profiles: dict[str, tuple[float, float]] | None = None,
        recv: bool = True,
    ) -> None:
        self._require_connected()
        if self._ee_body is None:
            raise RuntimeError("No ee_body is configured for this OpenArm hardware config")
        self._ee_body.send_positions(
            arms=self._arms,
            positions=positions,
            recv_timeout_us=self.config.safety.recv_timeout_us,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            profiles=profiles,
            recv=recv,
        )

    def send_ee_from_controller_inputs(
        self,
        controllers: dict[str, Any],
        *,
        states: dict[str, MotorState] | None = None,
        now: float | None = None,
        recv: bool = True,
    ) -> dict[str, str]:
        self._require_connected()
        if self._ee_body is None:
            raise RuntimeError("No ee_body is configured for this OpenArm hardware config")
        state_snapshot = self.read_state(recv_timeout_us=self.config.safety.recv_timeout_us) if states is None else states
        return self._ee_body.send_from_controller_inputs(
            arms=self._arms,
            controllers=controllers,
            states=state_snapshot,
            now=0.0 if now is None else float(now),
            recv_timeout_us=self.config.safety.recv_timeout_us,
            recv=recv,
        )

    def send_gripper_positions(
        self,
        positions: dict[str, float],
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        profiles: dict[str, tuple[float, float]] | None = None,
        recv: bool = True,
    ) -> None:
        self.send_ee_positions(
            positions,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            profiles=profiles,
            recv=recv,
        )

    def set_ee_zero(self) -> None:
        self._require_connected()
        if self._ee_body is None:
            raise RuntimeError("No ee_body is configured for this OpenArm hardware config")
        self._ee_body.set_zero(
            arms=self._arms,
            recv_timeout_us=self.config.safety.enable_recv_timeout_us,
        )

    def _require_connected(self) -> None:
        if not self.connected or self.oa is None:
            raise RuntimeError("OpenArmCANBridge.connect() must be called before hardware access")


def _optional_float_call(obj: Any, method_name: str) -> float | None:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return float(method())
    except Exception:
        return None


def _optional_bool_call(obj: Any, method_name: str) -> bool | None:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return bool(method())
    except Exception:
        return None
