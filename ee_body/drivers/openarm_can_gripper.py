"""OpenArm CAN gripper end-effector body."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable


def _parse_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


@dataclass(frozen=True)
class GripperConfig:
    name: str
    side: str
    motor_type: str
    send_can_id: int
    recv_can_id: int
    control_mode: str
    open_position: float
    close_position: float
    safe_close_position: float
    open_speed_rad_s: float
    close_speed_rad_s: float
    open_torque_pu: float
    close_torque_pu: float
    torque_stop_threshold: float | None
    stall_velocity_threshold: float
    stall_hold_time_s: float
    trigger_deadband: float
    release_trigger_threshold: float
    kp: float
    kd: float
    sign: float
    zero_offset: float
    enabled: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GripperConfig":
        speed_rad_s = float(payload.get("speed_rad_s", 5.0))
        torque_pu = float(payload.get("torque_pu", 0.5))
        close_position = float(payload.get("close_position", 0.0))
        return cls(
            name=str(payload["name"]),
            side=str(payload["side"]),
            motor_type=str(payload["motor_type"]),
            send_can_id=_parse_int(payload["send_can_id"]),
            recv_can_id=_parse_int(payload["recv_can_id"]),
            control_mode=str(payload.get("control_mode", "POS_FORCE")),
            open_position=float(payload.get("open_position", 1.0)),
            close_position=close_position,
            safe_close_position=float(payload.get("safe_close_position", close_position)),
            open_speed_rad_s=float(payload.get("open_speed_rad_s", speed_rad_s)),
            close_speed_rad_s=float(payload.get("close_speed_rad_s", speed_rad_s)),
            open_torque_pu=float(payload.get("open_torque_pu", torque_pu)),
            close_torque_pu=float(payload.get("close_torque_pu", torque_pu)),
            torque_stop_threshold=(
                None if payload.get("torque_stop_threshold") is None else float(payload["torque_stop_threshold"])
            ),
            stall_velocity_threshold=float(payload.get("stall_velocity_threshold", 0.02)),
            stall_hold_time_s=float(payload.get("stall_hold_time_s", 0.15)),
            trigger_deadband=float(payload.get("trigger_deadband", 0.02)),
            release_trigger_threshold=float(payload.get("release_trigger_threshold", 0.03)),
            kp=float(payload.get("kp", 50.0)),
            kd=float(payload.get("kd", 1.0)),
            sign=float(payload.get("sign", 1.0)),
            zero_offset=float(payload.get("zero_offset", 0.0)),
            enabled=bool(payload.get("enabled", True)),
        )

    def logical_to_hardware(self, position: float) -> float:
        return float(self.sign) * float(position) + float(self.zero_offset)

    def hardware_to_logical(self, position: float) -> float:
        if abs(float(self.sign)) < 1e-8:
            raise ValueError(f"Gripper {self.name} has invalid sign={self.sign}")
        return (float(position) - float(self.zero_offset)) / float(self.sign)

    @property
    def close_direction(self) -> float:
        return 1.0 if self.safe_close_position >= self.open_position else -1.0

    def trigger_to_position(self, trigger: float) -> float:
        deadband = min(max(float(self.trigger_deadband), 0.0), 0.95)
        value = min(max(float(trigger), 0.0), 1.0)
        if value <= deadband:
            ratio = 0.0
        else:
            ratio = (value - deadband) / (1.0 - deadband)
        return self.open_position + ratio * (self.safe_close_position - self.open_position)

    def is_closing(self, current: float, target: float) -> bool:
        return (float(target) - float(current)) * self.close_direction > 1e-5


class OpenArmCANGripperEE:
    def __init__(
        self,
        *,
        config_path: Path,
        name: str,
        enabled: bool,
        grippers: tuple[GripperConfig, ...],
    ) -> None:
        self.config_path = config_path
        self.name = name
        self.enabled = bool(enabled)
        self.grippers = tuple(gripper for gripper in grippers if gripper.enabled) if self.enabled else ()
        self._hold_positions: dict[str, float] = {}
        self._stall_since: dict[str, float] = {}

    @classmethod
    def from_config(cls, config_path: Path, payload: dict[str, Any]) -> "OpenArmCANGripperEE":
        return cls(
            config_path=config_path,
            name=str(payload.get("name", config_path.stem)),
            enabled=bool(payload.get("enabled", True)),
            grippers=tuple(GripperConfig.from_payload(item) for item in payload.get("grippers", [])),
        )

    def grippers_for_side(self, side: str) -> tuple[GripperConfig, ...]:
        return tuple(gripper for gripper in self.grippers if gripper.side == side)

    def gripper_by_name(self, name: str) -> GripperConfig:
        for gripper in self.grippers:
            if gripper.name == name:
                return gripper
        raise KeyError(f"Unknown gripper: {name}")

    def update_gripper_config(self, name: str, **changes: Any) -> GripperConfig:
        updated = replace(self.gripper_by_name(name), **changes)
        self.grippers = tuple(updated if gripper.name == name else gripper for gripper in self.grippers)
        self._hold_positions.pop(name, None)
        self._stall_since.pop(name, None)
        return updated

    def save_config(self) -> None:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        gripper_by_name = {gripper.name: gripper for gripper in self.grippers}
        for item in payload.get("grippers", []):
            gripper = gripper_by_name.get(str(item.get("name", "")))
            if gripper is None:
                continue
            item["safe_close_position"] = float(gripper.safe_close_position)
            item["open_speed_rad_s"] = float(gripper.open_speed_rad_s)
            item["close_speed_rad_s"] = float(gripper.close_speed_rad_s)
            item["open_torque_pu"] = float(gripper.open_torque_pu)
            item["close_torque_pu"] = float(gripper.close_torque_pu)
            item["torque_stop_threshold"] = (
                None if gripper.torque_stop_threshold is None else float(gripper.torque_stop_threshold)
            )
            item["sign"] = float(gripper.sign)
        self.config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def init_bus(
        self,
        *,
        side: str,
        arm: Any,
        motor_type: Callable[[str], Any],
        control_mode: Callable[[str], Any],
    ) -> None:
        for gripper in self.grippers_for_side(side):
            arm.init_gripper_motor(
                motor_type(gripper.motor_type),
                gripper.send_can_id,
                gripper.recv_can_id,
                control_mode(gripper.control_mode),
            )

    def read_state(
        self,
        *,
        side: str,
        arm: Any,
        make_state: Callable[..., Any],
        optional_float: Callable[[Any, str], float | None],
        optional_bool: Callable[[Any, str], bool | None],
    ) -> dict[str, Any]:
        states: dict[str, Any] = {}
        grippers = self.grippers_for_side(side)
        if not grippers:
            return states
        for gripper, motor in zip(grippers, arm.get_gripper().get_motors(), strict=False):
            states[gripper.name] = make_state(
                joint_name=gripper.name,
                position=gripper.hardware_to_logical(float(motor.get_position())),
                velocity=optional_float(motor, "get_velocity"),
                torque=optional_float(motor, "get_torque"),
                enabled=optional_bool(motor, "is_enabled"),
            )
        return states

    def send_positions(
        self,
        *,
        arms: dict[str, Any],
        positions: dict[str, float],
        recv_timeout_us: int,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        profiles: dict[str, tuple[float, float]] | None = None,
        recv: bool = True,
    ) -> None:
        profiles = {} if profiles is None else profiles
        for side, arm in arms.items():
            grippers = self.grippers_for_side(side)
            if not grippers:
                continue
            component = arm.get_gripper()
            for gripper in grippers:
                if gripper.name not in positions:
                    continue
                target = gripper.logical_to_hardware(float(positions[gripper.name]))
                if gripper.control_mode.upper() == "POS_FORCE":
                    speed_rad_s, torque_pu = profiles.get(
                        gripper.name,
                        (gripper.close_speed_rad_s, gripper.close_torque_pu),
                    )
                    component.set_position(
                        target,
                        speed_rad_s=float(speed_rad_s),
                        torque_pu=float(torque_pu),
                    )
                else:
                    component.set_position_mit(
                        target,
                        kp=float(gripper.kp) * float(kp_scale),
                        kd=float(gripper.kd) * float(kd_scale),
                    )
            if recv:
                arm.recv_all(recv_timeout_us)

    def set_zero(self, *, arms: dict[str, Any], recv_timeout_us: int) -> None:
        for side, arm in arms.items():
            if not self.grippers_for_side(side):
                continue
            arm.get_gripper().set_zero()
            arm.recv_all(recv_timeout_us)

    def create_tuner_panel(
        self,
        *,
        parent: Any,
        bridge: Any,
        status_callback: Callable[[str], None],
        hz: float,
    ) -> Any:
        return OpenArmCANGripperTunerPanel(
            parent=parent,
            ee_body=self,
            bridge=bridge,
            status_callback=status_callback,
            hz=hz,
        )

    def targets_from_controller_inputs(
        self,
        *,
        controllers: dict[str, Any],
        states: dict[str, Any],
        now: float,
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]], dict[str, str]]:
        positions: dict[str, float] = {}
        profiles: dict[str, tuple[float, float]] = {}
        statuses: dict[str, str] = {}
        for gripper in self.grippers:
            trigger = self._controller_trigger(controllers, gripper.side)
            state = states.get(gripper.name)
            current = (
                float(getattr(state, "position"))
                if state is not None and getattr(state, "position", None) is not None
                else self._hold_positions.get(gripper.name, gripper.open_position)
            )
            requested = gripper.trigger_to_position(trigger)

            if trigger <= gripper.release_trigger_threshold:
                self._hold_positions.pop(gripper.name, None)
                self._stall_since.pop(gripper.name, None)

            hold_position = self._hold_positions.get(gripper.name)
            closing = gripper.is_closing(current, requested)
            if hold_position is None and closing and self._should_hold_contact(gripper, state, current, requested, now):
                hold_position = current
                self._hold_positions[gripper.name] = hold_position

            if hold_position is not None:
                positions[gripper.name] = hold_position
                profiles[gripper.name] = (gripper.close_speed_rad_s, gripper.close_torque_pu)
                statuses[gripper.name] = f"holding_contact pos={hold_position:.4f}"
            else:
                positions[gripper.name] = requested
                if closing:
                    profiles[gripper.name] = (gripper.close_speed_rad_s, gripper.close_torque_pu)
                    statuses[gripper.name] = "closing"
                else:
                    profiles[gripper.name] = (gripper.open_speed_rad_s, gripper.open_torque_pu)
                    statuses[gripper.name] = "opening"
        return positions, profiles, statuses

    def send_from_controller_inputs(
        self,
        *,
        arms: dict[str, Any],
        controllers: dict[str, Any],
        states: dict[str, Any],
        now: float,
        recv_timeout_us: int,
        recv: bool = True,
    ) -> dict[str, str]:
        positions, profiles, statuses = self.targets_from_controller_inputs(
            controllers=controllers,
            states=states,
            now=now,
        )
        self.send_positions(
            arms=arms,
            positions=positions,
            recv_timeout_us=recv_timeout_us,
            profiles=profiles,
            recv=recv,
        )
        return statuses

    @staticmethod
    def _controller_trigger(controllers: dict[str, Any], side: str) -> float:
        controller = controllers.get(side, {})
        if isinstance(controller, dict):
            return float(controller.get("trigger", 0.0) or 0.0)
        return float(getattr(controller, "trigger", 0.0) or 0.0)

    def _should_hold_contact(
        self,
        gripper: GripperConfig,
        state: Any,
        current: float,
        requested: float,
        now: float,
    ) -> bool:
        if state is None:
            return False

        torque = getattr(state, "torque", None)
        if gripper.torque_stop_threshold is not None and torque is not None:
            if abs(float(torque)) >= float(gripper.torque_stop_threshold):
                return True

        velocity = getattr(state, "velocity", None)
        if velocity is None:
            self._stall_since.pop(gripper.name, None)
            return False

        still_moving_to_target = abs(float(requested) - float(current)) > 0.02
        stalled = abs(float(velocity)) <= float(gripper.stall_velocity_threshold)
        if stalled and still_moving_to_target:
            start = self._stall_since.setdefault(gripper.name, now)
            return now - start >= float(gripper.stall_hold_time_s)

        self._stall_since.pop(gripper.name, None)
        return False


class OpenArmCANGripperTunerPanel:
    """Tk panel owned by the gripper plugin, mounted by the generic hardware tuner."""

    def __init__(
        self,
        *,
        parent: Any,
        ee_body: OpenArmCANGripperEE,
        bridge: Any,
        status_callback: Callable[[str], None],
        hz: float,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.parent = parent
        self.ee_body = ee_body
        self.bridge = bridge
        self.status_callback = status_callback
        self.hz = float(hz)
        self.motors_enabled = False
        self.last_states: dict[str, Any] = {}
        self.initialized_from_feedback: set[str] = set()

        self.target_vars: dict[str, Any] = {}
        self.sign_vars: dict[str, Any] = {}
        self.safe_close_vars: dict[str, Any] = {}
        self.open_speed_vars: dict[str, Any] = {}
        self.close_speed_vars: dict[str, Any] = {}
        self.open_torque_vars: dict[str, Any] = {}
        self.close_torque_vars: dict[str, Any] = {}
        self.torque_stop_vars: dict[str, Any] = {}
        self.feedback_vars: dict[str, Any] = {}
        self.enabled_vars: dict[str, Any] = {}
        self.status_vars: dict[str, Any] = {}
        self.scale_widgets: dict[str, Any] = {}
        self.auto_send_var = tk.BooleanVar(value=False)

        self._build()

    def _build(self) -> None:
        toolbar = self.ttk.Frame(self.parent)
        toolbar.grid(row=0, column=0, columnspan=12, sticky="ew", pady=(0, 8))
        self.ttk.Button(toolbar, text="Send All EE", command=self.send_all).grid(row=0, column=0, padx=(0, 6))
        self.ttk.Checkbutton(toolbar, text="Auto Send EE", variable=self.auto_send_var).grid(row=0, column=1, padx=(0, 12))
        self.ttk.Button(toolbar, text="Set EE Zero", command=self.set_zero).grid(row=0, column=2, padx=(0, 6))
        self.ttk.Button(toolbar, text="Save EE JSON", command=self.save_json).grid(row=0, column=3, padx=(0, 6))

        headings = [
            "Name",
            "Side",
            "Feedback p/v/t",
            "Enabled",
            "Sign",
            "Safe close",
            "Open v/t",
            "Close v/t",
            "Torque stop",
            "Target",
            "Slider",
            "Status",
            "Commands",
        ]
        for col, heading in enumerate(headings):
            self.ttk.Label(self.parent, text=heading).grid(row=1, column=col, sticky="w", padx=4, pady=(0, 6))

        for row, gripper in enumerate(self.ee_body.grippers, start=2):
            lower = min(gripper.open_position, gripper.safe_close_position, gripper.close_position)
            upper = max(gripper.open_position, gripper.safe_close_position, gripper.close_position)
            target_var = self.tk.DoubleVar(value=float(gripper.open_position))
            sign_var = self.tk.StringVar(value=f"{float(gripper.sign):+.0f}")
            safe_close_var = self.tk.DoubleVar(value=float(gripper.safe_close_position))
            open_speed_var = self.tk.DoubleVar(value=float(gripper.open_speed_rad_s))
            close_speed_var = self.tk.DoubleVar(value=float(gripper.close_speed_rad_s))
            open_torque_var = self.tk.DoubleVar(value=float(gripper.open_torque_pu))
            close_torque_var = self.tk.DoubleVar(value=float(gripper.close_torque_pu))
            torque_stop_var = self.tk.StringVar(
                value="" if gripper.torque_stop_threshold is None else f"{float(gripper.torque_stop_threshold):.4f}"
            )
            self.target_vars[gripper.name] = target_var
            self.sign_vars[gripper.name] = sign_var
            self.safe_close_vars[gripper.name] = safe_close_var
            self.open_speed_vars[gripper.name] = open_speed_var
            self.close_speed_vars[gripper.name] = close_speed_var
            self.open_torque_vars[gripper.name] = open_torque_var
            self.close_torque_vars[gripper.name] = close_torque_var
            self.torque_stop_vars[gripper.name] = torque_stop_var
            self.feedback_vars[gripper.name] = self.tk.StringVar(value="--")
            self.enabled_vars[gripper.name] = self.tk.StringVar(value="--")
            self.status_vars[gripper.name] = self.tk.StringVar(value="idle")

            self.ttk.Label(self.parent, text=gripper.name).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            self.ttk.Label(self.parent, text=gripper.side).grid(row=row, column=1, sticky="w", padx=4, pady=3)
            self.ttk.Label(self.parent, textvariable=self.feedback_vars[gripper.name], width=22).grid(row=row, column=2, sticky="w", padx=4)
            self.ttk.Label(self.parent, textvariable=self.enabled_vars[gripper.name], width=8).grid(row=row, column=3, sticky="w", padx=4)
            self.ttk.Entry(self.parent, textvariable=sign_var, width=5).grid(row=row, column=4, sticky="w", padx=4)
            self.ttk.Entry(self.parent, textvariable=safe_close_var, width=9).grid(row=row, column=5, sticky="w", padx=4)
            open_profile = self.ttk.Frame(self.parent)
            open_profile.grid(row=row, column=6, sticky="w", padx=4)
            self.ttk.Entry(open_profile, textvariable=open_speed_var, width=6).grid(row=0, column=0)
            self.ttk.Label(open_profile, text="/").grid(row=0, column=1)
            self.ttk.Entry(open_profile, textvariable=open_torque_var, width=6).grid(row=0, column=2)
            close_profile = self.ttk.Frame(self.parent)
            close_profile.grid(row=row, column=7, sticky="w", padx=4)
            self.ttk.Entry(close_profile, textvariable=close_speed_var, width=6).grid(row=0, column=0)
            self.ttk.Label(close_profile, text="/").grid(row=0, column=1)
            self.ttk.Entry(close_profile, textvariable=close_torque_var, width=6).grid(row=0, column=2)
            self.ttk.Entry(self.parent, textvariable=torque_stop_var, width=10).grid(row=row, column=8, sticky="w", padx=4)
            self.ttk.Entry(self.parent, textvariable=target_var, width=10).grid(row=row, column=9, sticky="w", padx=4)
            scale = self.tk.Scale(
                self.parent,
                from_=float(lower),
                to=float(upper),
                orient=self.tk.HORIZONTAL,
                resolution=0.001,
                length=260,
                variable=target_var,
                showvalue=False,
            )
            self.scale_widgets[gripper.name] = scale
            scale.grid(row=row, column=10, sticky="ew", padx=4)
            self.ttk.Label(self.parent, textvariable=self.status_vars[gripper.name], width=18).grid(row=row, column=11, sticky="w", padx=4)

            buttons = self.ttk.Frame(self.parent)
            buttons.grid(row=row, column=12, sticky="w", padx=4)
            self.ttk.Button(
                buttons,
                text="Open",
                command=lambda item=gripper: self.command_preset(item, "open"),
            ).grid(row=0, column=0, padx=(0, 4))
            self.ttk.Button(
                buttons,
                text="Safe Close",
                command=lambda item=gripper: self.command_preset(item, "safe_close"),
            ).grid(row=0, column=1, padx=(0, 4))
            self.ttk.Button(
                buttons,
                text="Close Limit",
                command=lambda item=gripper: self.command_preset(item, "close_limit"),
            ).grid(row=0, column=2, padx=(0, 4))
            self.ttk.Button(
                buttons,
                text="Send",
                command=lambda item=gripper: self.send_one(item.name),
            ).grid(row=0, column=3, padx=(0, 4))
            self.ttk.Button(
                buttons,
                text="Flip Sign",
                command=lambda item=gripper: self.flip_sign(item.name),
            ).grid(row=0, column=4)

        self.parent.columnconfigure(10, weight=1)

    def update_state(self, states: dict[str, Any]) -> None:
        self.last_states = dict(states)
        for gripper in self.ee_body.grippers:
            state = states.get(gripper.name)
            if state is None:
                self.actual_vars[gripper.name].set("--")
                self.velocity_vars[gripper.name].set("--")
                self.torque_feedback_vars[gripper.name].set("--")
                self.enabled_vars[gripper.name].set("--")
                continue

            position = float(getattr(state, "position"))
            velocity = self._format_optional(getattr(state, "velocity", None))
            torque = self._format_optional(getattr(state, "torque", None))
            self.feedback_vars[gripper.name].set(f"{position:.4f}/{velocity}/{torque}")
            enabled = getattr(state, "enabled", None)
            self.enabled_vars[gripper.name].set("?" if enabled is None else ("yes" if enabled else "no"))
            if gripper.name not in self.initialized_from_feedback:
                self.target_vars[gripper.name].set(position)
                self.initialized_from_feedback.add(gripper.name)

        if self.motors_enabled and bool(self.auto_send_var.get()):
            self.send_all(status_prefix="auto sent")

    def on_motors_enabled(self) -> None:
        self.motors_enabled = True

    def on_motors_disabled(self) -> None:
        self.motors_enabled = False
        self.auto_send_var.set(False)

    def close(self) -> None:
        self.auto_send_var.set(False)

    def command_preset(self, gripper: GripperConfig, preset: str) -> None:
        self._apply_vars_to_runtime()
        gripper = self.ee_body.gripper_by_name(gripper.name)
        if preset == "open":
            self.target_vars[gripper.name].set(float(gripper.open_position))
        elif preset == "safe_close":
            self.target_vars[gripper.name].set(float(gripper.safe_close_position))
        elif preset == "close_limit":
            ok = self.messagebox.askyesno(
                "Close Limit",
                f"Send {gripper.name} to mechanical close_position={gripper.close_position:.4f}? "
                "Use Safe Close for normal grasping.",
            )
            if not ok:
                return
            self.target_vars[gripper.name].set(float(gripper.close_position))
        else:
            raise ValueError(f"Unsupported gripper preset: {preset}")
        self.send_one(gripper.name)

    def send_one(self, name: str) -> None:
        self._apply_vars_to_runtime()
        self._send_positions({name: self._target(name)}, {name: self._profile(name)}, status_prefix="sent")

    def send_all(self, *, status_prefix: str = "sent") -> None:
        self._apply_vars_to_runtime()
        positions = {gripper.name: self._target(gripper.name) for gripper in self.ee_body.grippers}
        profiles = {gripper.name: self._profile(gripper.name) for gripper in self.ee_body.grippers}
        self._send_positions(positions, profiles, status_prefix=status_prefix)

    def set_zero(self) -> None:
        ok = self.messagebox.askyesno(
            "Set EE Zero",
            "This sends set_zero() to the configured end-effector motors. Continue?",
        )
        if not ok:
            return
        try:
            self.bridge.set_ee_zero()
            self.status_callback("ee zero command sent")
        except Exception as exc:
            self.status_callback(f"ee zero failed: {exc}")

    def save_json(self) -> None:
        try:
            self._apply_vars_to_runtime()
            self.ee_body.save_config()
            self.status_callback(f"saved ee config: {self.ee_body.config_path}")
        except Exception as exc:
            self.status_callback(f"save ee config failed: {exc}")

    def flip_sign(self, name: str) -> None:
        try:
            current = self._sign(name)
            new_sign = -1.0 if current >= 0.0 else 1.0
            self.sign_vars[name].set(f"{new_sign:+.0f}")
            self._apply_vars_to_runtime()
            self.initialized_from_feedback.discard(name)
            self.status_callback(f"flipped {name} ee sign to {new_sign:+.0f}; click Save EE JSON to persist")
        except Exception as exc:
            self.status_callback(f"flip ee sign failed: {exc}")

    def _send_positions(
        self,
        positions: dict[str, float],
        profiles: dict[str, tuple[float, float]],
        *,
        status_prefix: str,
    ) -> None:
        if not self.motors_enabled:
            self.status_callback("motors are disabled; click Enable All first")
            return
        try:
            self.bridge.send_ee_positions(positions, profiles=profiles)
            for name, target in positions.items():
                speed, torque = profiles[name]
                self.status_vars[name].set(f"{target:.3f} v={speed:.2f} t={torque:.2f}")
            self.status_callback(f"{status_prefix} ee target")
        except Exception as exc:
            self.status_callback(f"ee send failed: {exc}")

    def _target(self, name: str) -> float:
        return float(self.target_vars[name].get())

    def _profile(self, name: str) -> tuple[float, float]:
        gripper = self.ee_body.gripper_by_name(name)
        state = self.last_states.get(name)
        current = (
            float(getattr(state, "position"))
            if state is not None and getattr(state, "position", None) is not None
            else self._target(name)
        )
        if gripper.is_closing(current, self._target(name)):
            return float(gripper.close_speed_rad_s), float(gripper.close_torque_pu)
        return float(gripper.open_speed_rad_s), float(gripper.open_torque_pu)

    def _apply_vars_to_runtime(self) -> None:
        for gripper in tuple(self.ee_body.grippers):
            updated = self.ee_body.update_gripper_config(
                gripper.name,
                safe_close_position=float(self.safe_close_vars[gripper.name].get()),
                open_speed_rad_s=float(self.open_speed_vars[gripper.name].get()),
                close_speed_rad_s=float(self.close_speed_vars[gripper.name].get()),
                open_torque_pu=float(self.open_torque_vars[gripper.name].get()),
                close_torque_pu=float(self.close_torque_vars[gripper.name].get()),
                torque_stop_threshold=self._torque_stop_threshold(gripper.name),
                sign=self._sign(gripper.name),
            )
            lower = min(updated.open_position, updated.safe_close_position, updated.close_position)
            upper = max(updated.open_position, updated.safe_close_position, updated.close_position)
            self.scale_widgets[gripper.name].configure(from_=float(lower), to=float(upper))

    def _sign(self, name: str) -> float:
        sign = float(self.sign_vars[name].get())
        if abs(sign) < 1e-8:
            raise ValueError(f"{name} sign cannot be zero")
        return 1.0 if sign > 0.0 else -1.0

    def _torque_stop_threshold(self, name: str) -> float | None:
        text = str(self.torque_stop_vars[name].get()).strip()
        if text.lower() in {"", "none", "null"}:
            return None
        return float(text)

    @staticmethod
    def _format_optional(value: Any) -> str:
        if value is None:
            return "--"
        return f"{float(value):.4f}"
