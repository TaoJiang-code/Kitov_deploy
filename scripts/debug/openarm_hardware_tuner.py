#!/usr/bin/env python3
"""Interactive OpenArm CAN hardware tuner."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.hardware.openarm_can_bridge import (  # noqa: E402
    DEFAULT_OPENARM_HARDWARE_CONFIG,
    OpenArmCANBridge,
    OpenArmCommandLimiter,
    OpenArmHardwareConfig,
    OpenArmQposMapper,
    load_openarm_hardware_config,
)
from kitov_deploy.gmr_online import make_robot_motion_viewer  # noqa: E402

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on system Tk install.
    tk = None
    ttk = None
    messagebox = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune OpenArm hardware joints with sliders.")
    parser.add_argument("--hardware-config", type=Path, default=DEFAULT_OPENARM_HARDWARE_CONFIG)
    parser.add_argument("--hz", type=float, default=50.0, help="Read/send refresh rate.")
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--slider-space",
        choices=["sim", "hardware"],
        default="sim",
        help="Interpret sliders as MuJoCo joint angles or raw hardware motor angles.",
    )
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True, help="Open MuJoCo viewer.")
    parser.add_argument("--disable-on-exit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _format_id(value: int) -> str:
    return f"0x{int(value):02x}"


def _clamp(value: float, lower: float | None, upper: float | None) -> float:
    result = float(value)
    if lower is not None:
        result = max(result, float(lower))
    if upper is not None:
        result = min(result, float(upper))
    return result


def _midpoint(lower: float | None, upper: float | None) -> float:
    if lower is not None and upper is not None:
        return 0.5 * (float(lower) + float(upper))
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_zero_offsets_to_config(path: Path, states: dict[str, Any], config: OpenArmHardwareConfig) -> None:
    payload = _read_json(path)
    state_positions = {name: float(state.position) for name, state in states.items()}
    for bus in payload.get("buses", []):
        for motor in bus.get("motors", []):
            joint_name = str(motor.get("joint_name", ""))
            if joint_name in state_positions:
                motor["zero_offset"] = state_positions[joint_name]
    _write_json(path, payload)


def _update_motor_json_field(path: Path, joint_name: str, field_name: str, value: Any) -> None:
    payload = _read_json(path)
    found = False
    for bus in payload.get("buses", []):
        for motor in bus.get("motors", []):
            if str(motor.get("joint_name", "")) == joint_name:
                motor[field_name] = value
                found = True
    if not found:
        raise KeyError(f"Cannot find joint in hardware config: {joint_name}")
    _write_json(path, payload)


class OpenArmHardwareTuner:
    def __init__(
        self,
        *,
        hardware_config_path: Path,
        hardware_config: OpenArmHardwareConfig,
        mapper: OpenArmQposMapper,
        bridge: OpenArmCANBridge,
        hz: float,
        kp_scale: float,
        kd_scale: float,
        slider_space: str,
        viewer: Any | None,
        disable_on_exit: bool,
    ) -> None:
        if tk is None or ttk is None or messagebox is None:
            raise RuntimeError(f"Tkinter is unavailable: {TK_IMPORT_ERROR}")
        self.hardware_config_path = hardware_config_path
        self.config = hardware_config
        self.mapper = mapper
        self.bridge = bridge
        self.period_ms = max(int(round(1000.0 / max(float(hz), 1e-6))), 1)
        self.kp_scale = float(kp_scale)
        self.kd_scale = float(kd_scale)
        self.slider_space = str(slider_space)
        self.viewer = viewer
        self.disable_on_exit = bool(disable_on_exit)
        self.limiter = OpenArmCommandLimiter(hardware_config)
        self.hardware_limits = mapper.hardware_limits()
        self.sim_limits = mapper.sim_joint_limits()
        self.slider_limits = dict(self.sim_limits if self.slider_space == "sim" else self.hardware_limits)
        self.motor_by_joint = {motor.joint_name: motor for motor in hardware_config.motors}

        self.motors_enabled = False
        self.auto_send_var: Any = None
        self.closed = False
        self.last_update_time = time.monotonic()
        self.last_sent_targets: dict[str, float] | None = None
        self.target_vars: dict[str, Any] = {}
        self.current_vars: dict[str, Any] = {}
        self.current_sim_vars: dict[str, Any] = {}
        self.sign_vars: dict[str, Any] = {}
        self.limit_vars: dict[str, Any] = {}
        self.scale_widgets: dict[str, Any] = {}
        self.sent_vars: dict[str, Any] = {}
        self.error_vars: dict[str, Any] = {}
        self.enabled_vars: dict[str, Any] = {}

        self.root = tk.Tk()
        self.root.title("OpenArm Hardware Tuner")
        self.root.geometry("1320x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="connecting")
        self._build_ui()
        self._initialize_targets_from_state()
        self._schedule_tick()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(8, weight=1)
        ttk.Button(header, text="Enable All", command=self.enable_all).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(header, text="Disable All", command=self.disable_all).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(header, text="Hold Current", command=self.hold_current).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(header, text="Send Once", command=self.send_once).grid(row=0, column=3, padx=(0, 6))
        self.auto_send_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text="Auto Send", variable=self.auto_send_var).grid(row=0, column=4, padx=(0, 14))
        ttk.Button(header, text="Set Motor Zero All", command=self.set_motor_zero_all).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(header, text="Save JSON Zero Offset", command=self.save_json_zero_offsets).grid(row=0, column=6, padx=(0, 14))
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=8, sticky="w")

        canvas = tk.Canvas(self.root, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        table = ttk.Frame(canvas, padding=(8, 0, 8, 8))
        table.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=table, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")

        headings = [
            "Joint",
            "Bus/IDs",
            "Slider limit rad",
            "Actual hw",
            "Actual sim",
            "Sign",
            "Target sim" if self.slider_space == "sim" else "Target hw",
            "Slider",
            "Sent hw",
            "Error",
            "Enabled",
            "Config",
        ]
        for col, heading in enumerate(headings):
            ttk.Label(table, text=heading).grid(row=0, column=col, sticky="w", padx=4, pady=(0, 8))

        for row, motor in enumerate(self.config.motors, start=1):
            lower, upper = self.slider_limits.get(motor.joint_name, (None, None))
            target_var = tk.DoubleVar(value=_midpoint(lower, upper))
            self.target_vars[motor.joint_name] = target_var
            self.current_vars[motor.joint_name] = tk.StringVar(value="--")
            self.current_sim_vars[motor.joint_name] = tk.StringVar(value="--")
            self.sign_vars[motor.joint_name] = tk.StringVar(value=f"{float(motor.sign):+.0f}")
            self.limit_vars[motor.joint_name] = tk.StringVar(value=self._format_limit(lower, upper))
            self.sent_vars[motor.joint_name] = tk.StringVar(value="--")
            self.error_vars[motor.joint_name] = tk.StringVar(value="--")
            self.enabled_vars[motor.joint_name] = tk.StringVar(value="--")

            ttk.Label(table, text=motor.joint_name).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Label(
                table,
                text=f"{motor.side} {_format_id(motor.send_can_id)}->{_format_id(motor.recv_can_id)}",
            ).grid(row=row, column=1, sticky="w", padx=4, pady=3)
            ttk.Label(table, textvariable=self.limit_vars[motor.joint_name]).grid(row=row, column=2, sticky="w", padx=4, pady=3)
            ttk.Label(table, textvariable=self.current_vars[motor.joint_name], width=10).grid(row=row, column=3, sticky="w", padx=4)
            ttk.Label(table, textvariable=self.current_sim_vars[motor.joint_name], width=10).grid(row=row, column=4, sticky="w", padx=4)
            ttk.Label(table, textvariable=self.sign_vars[motor.joint_name], width=5).grid(row=row, column=5, sticky="w", padx=4)
            ttk.Entry(table, textvariable=target_var, width=11).grid(row=row, column=6, sticky="w", padx=4)
            scale = tk.Scale(
                table,
                from_=float(lower if lower is not None else -np.pi),
                to=float(upper if upper is not None else np.pi),
                orient=tk.HORIZONTAL,
                resolution=0.001,
                length=360,
                variable=target_var,
                showvalue=False,
            )
            self.scale_widgets[motor.joint_name] = scale
            scale.grid(row=row, column=7, sticky="ew", padx=4)
            ttk.Label(table, textvariable=self.sent_vars[motor.joint_name], width=10).grid(row=row, column=8, sticky="w", padx=4)
            ttk.Label(table, textvariable=self.error_vars[motor.joint_name], width=10).grid(row=row, column=9, sticky="w", padx=4)
            ttk.Label(table, textvariable=self.enabled_vars[motor.joint_name], width=8).grid(row=row, column=10, sticky="w", padx=4)
            ttk.Button(
                table,
                text="Flip Sign",
                command=lambda joint_name=motor.joint_name: self.flip_sign(joint_name),
            ).grid(row=row, column=11, sticky="w", padx=4)

        table.columnconfigure(7, weight=1)

    @staticmethod
    def _format_limit(lower: float | None, upper: float | None) -> str:
        return f"[{('-inf' if lower is None else f'{lower:.3f}')}, {('inf' if upper is None else f'{upper:.3f}')} ]"

    def _reload_config_mapping(self) -> None:
        self.config = load_openarm_hardware_config(self.hardware_config_path)
        self.mapper = OpenArmQposMapper(self.config)
        self.limiter = OpenArmCommandLimiter(self.config)
        self.hardware_limits = self.mapper.hardware_limits()
        self.sim_limits = self.mapper.sim_joint_limits()
        self.motor_by_joint = {motor.joint_name: motor for motor in self.config.motors}
        current_targets = self._hardware_targets_from_slider_values()
        self.limiter.reset(current_targets)
        self.last_sent_targets = dict(current_targets)
        for motor in self.config.motors:
            lower, upper = self.slider_limits.get(motor.joint_name, (None, None))
            self.sign_vars[motor.joint_name].set(f"{float(motor.sign):+.0f}")
            self.limit_vars[motor.joint_name].set(self._format_limit(lower, upper))
            self.target_vars[motor.joint_name].set(_clamp(float(self.target_vars[motor.joint_name].get()), lower, upper))
            scale = self.scale_widgets.get(motor.joint_name)
            if scale is not None:
                scale.configure(
                    from_=float(lower if lower is not None else -np.pi),
                    to=float(upper if upper is not None else np.pi),
                )

    def _initialize_targets_from_state(self) -> None:
        states = self.bridge.read_state(recv_timeout_us=self.config.safety.enable_recv_timeout_us)
        targets = {}
        for motor in self.config.motors:
            lower, upper = self.slider_limits.get(motor.joint_name, (None, None))
            value = _midpoint(lower, upper)
            if motor.joint_name in states:
                hardware_q = float(states[motor.joint_name].position)
                value = motor.hardware_to_sim(hardware_q) if self.slider_space == "sim" else hardware_q
            value = _clamp(value, lower, upper)
            self.target_vars[motor.joint_name].set(value)
        targets = self._hardware_targets_from_slider_values()
        self.limiter.reset(targets)
        self.last_sent_targets = dict(targets)
        self._update_state_labels(states)
        self.status_var.set(f"connected; {self.slider_space} sliders initialized from current feedback")

    def _target_values(self) -> dict[str, float]:
        targets = {}
        for motor in self.config.motors:
            lower, upper = self.slider_limits.get(motor.joint_name, (None, None))
            targets[motor.joint_name] = _clamp(float(self.target_vars[motor.joint_name].get()), lower, upper)
        return targets

    def _hardware_targets_from_slider_values(self) -> dict[str, float]:
        values = self._target_values()
        if self.slider_space == "hardware":
            targets = dict(values)
        else:
            qpos = self.mapper.qpos_from_sim_joint_positions(np.zeros(self.mapper.model.nq, dtype=np.float64), values)
            targets = self.mapper.hardware_targets_from_qpos(qpos)
        clipped = {}
        for motor in self.config.motors:
            lower, upper = self.hardware_limits.get(motor.joint_name, (None, None))
            clipped[motor.joint_name] = _clamp(targets[motor.joint_name], lower, upper)
        return clipped

    def _qpos_from_slider_values(self) -> np.ndarray:
        values = self._target_values()
        if self.slider_space == "sim":
            return self.mapper.qpos_from_sim_joint_positions(np.zeros(self.mapper.model.nq, dtype=np.float64), values)
        return self.mapper.qpos_from_hardware_targets(np.zeros(self.mapper.model.nq, dtype=np.float64), values)

    def _limited_targets(self) -> dict[str, float]:
        now = time.monotonic()
        dt = max(now - self.last_update_time, self.period_ms / 1000.0)
        self.last_update_time = now
        return self.limiter.limit(self._hardware_targets_from_slider_values(), dt)

    def _send_targets(self) -> None:
        targets = self._limited_targets()
        self.bridge.send_position_targets(targets, kp_scale=self.kp_scale, kd_scale=self.kd_scale)
        self.last_sent_targets = dict(targets)
        for joint_name, value in targets.items():
            self.sent_vars[joint_name].set(f"{value:.4f}")

    def _update_state_labels(self, states: dict[str, Any]) -> None:
        for motor in self.config.motors:
            state = states.get(motor.joint_name)
            if state is None:
                self.current_vars[motor.joint_name].set("--")
                self.current_sim_vars[motor.joint_name].set("--")
                self.error_vars[motor.joint_name].set("--")
                self.enabled_vars[motor.joint_name].set("--")
                continue
            actual = float(state.position)
            self.current_vars[motor.joint_name].set(f"{actual:.4f}")
            self.current_sim_vars[motor.joint_name].set(f"{motor.hardware_to_sim(actual):.4f}")
            enabled = "?" if state.enabled is None else ("yes" if state.enabled else "no")
            self.enabled_vars[motor.joint_name].set(enabled)
            if self.last_sent_targets is None or motor.joint_name not in self.last_sent_targets:
                self.error_vars[motor.joint_name].set("--")
            else:
                self.error_vars[motor.joint_name].set(f"{actual - self.last_sent_targets[motor.joint_name]:+.4f}")
        self._update_viewer()

    def _update_viewer(self) -> None:
        if self.viewer is None:
            return
        qpos = self._qpos_from_slider_values()
        try:
            self.viewer.step_qpos(qpos, rate_limit=False)
        except Exception as exc:
            self.status_var.set(f"viewer error: {exc}")

    def _tick(self) -> None:
        if self.closed:
            return
        try:
            if self.motors_enabled and bool(self.auto_send_var.get()):
                self._send_targets()
            states = self.bridge.read_state()
            self._update_state_labels(states)
        except Exception as exc:
            self.status_var.set(f"error: {exc}")
        self._schedule_tick()

    def _schedule_tick(self) -> None:
        if not self.closed:
            self.root.after(self.period_ms, self._tick)

    def enable_all(self) -> None:
        try:
            self.bridge.enable_all()
            self.motors_enabled = True
            self.hold_current()
            self.status_var.set("motors enabled; holding current target")
        except Exception as exc:
            self.status_var.set(f"enable failed: {exc}")

    def disable_all(self) -> None:
        try:
            self.bridge.disable_all()
            self.motors_enabled = False
            self.status_var.set("motors disabled")
        except Exception as exc:
            self.status_var.set(f"disable failed: {exc}")

    def hold_current(self) -> None:
        states = self.bridge.read_state(recv_timeout_us=self.config.safety.enable_recv_timeout_us)
        targets = {}
        for motor in self.config.motors:
            if motor.joint_name not in states:
                continue
            lower, upper = self.slider_limits.get(motor.joint_name, (None, None))
            hardware_q = float(states[motor.joint_name].position)
            value = motor.hardware_to_sim(hardware_q) if self.slider_space == "sim" else hardware_q
            value = _clamp(value, lower, upper)
            self.target_vars[motor.joint_name].set(value)
        targets = self._hardware_targets_from_slider_values()
        if targets:
            self.limiter.reset(targets)
            self.last_sent_targets = dict(targets)
        self._update_state_labels(states)
        self.status_var.set("target sliders copied from current feedback")

    def send_once(self) -> None:
        if not self.motors_enabled:
            self.status_var.set("motors are disabled; click Enable All first")
            return
        try:
            self._send_targets()
            self.status_var.set("sent one limited target step")
        except Exception as exc:
            self.status_var.set(f"send failed: {exc}")

    def set_motor_zero_all(self) -> None:
        ok = messagebox.askyesno(
            "Set Motor Zero All",
            "This sends set_zero_all() to the motors and changes their hardware zero. Continue?",
        )
        if not ok:
            return
        try:
            self.bridge.set_zero_all()
            time.sleep(0.2)
            self.hold_current()
            self.status_var.set("motor zero command sent; feedback refreshed")
        except Exception as exc:
            self.status_var.set(f"set zero failed: {exc}")

    def save_json_zero_offsets(self) -> None:
        ok = messagebox.askyesno(
            "Save JSON Zero Offset",
            f"Write current motor positions to zero_offset in {self.hardware_config_path}?",
        )
        if not ok:
            return
        try:
            states = self.bridge.read_state(recv_timeout_us=self.config.safety.enable_recv_timeout_us)
            _save_zero_offsets_to_config(self.hardware_config_path, states, self.config)
            self._reload_config_mapping()
            self._update_state_labels(states)
            self.status_var.set("saved current feedback as JSON zero_offset")
        except Exception as exc:
            self.status_var.set(f"save zero_offset failed: {exc}")

    def flip_sign(self, joint_name: str) -> None:
        motor = self.motor_by_joint[joint_name]
        new_sign = -1.0 if float(motor.sign) >= 0.0 else 1.0
        ok = messagebox.askyesno(
            "Flip Sign",
            f"Change {joint_name} sign from {motor.sign:+.0f} to {new_sign:+.0f} in {self.hardware_config_path}?",
        )
        if not ok:
            return
        try:
            _update_motor_json_field(self.hardware_config_path, joint_name, "sign", new_sign)
            states = self.bridge.read_state(recv_timeout_us=self.config.safety.enable_recv_timeout_us)
            self._reload_config_mapping()
            self._update_state_labels(states)
            self.status_var.set(f"flipped {joint_name} sign to {new_sign:+.0f}")
        except Exception as exc:
            self.status_var.set(f"flip sign failed: {exc}")

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.disable_on_exit and self.motors_enabled:
                self.bridge.disable_all()
        except Exception:
            pass
        try:
            if self.viewer is not None:
                self.viewer.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> int:
    args = _parse_args()
    hardware_config_path = args.hardware_config.expanduser()
    if not hardware_config_path.is_absolute():
        hardware_config_path = (REPO_ROOT / hardware_config_path).resolve(strict=False)
    hardware_config = load_openarm_hardware_config(hardware_config_path)
    mapper = OpenArmQposMapper(hardware_config)
    bridge = OpenArmCANBridge(hardware_config)
    print("[openarm_hardware_tuner] connecting to OpenArm CAN")
    bridge.connect()
    print("[openarm_hardware_tuner] connected")
    viewer = make_robot_motion_viewer("openarm_v1", motion_fps=args.hz) if args.viewer else None
    app = OpenArmHardwareTuner(
        hardware_config_path=hardware_config_path,
        hardware_config=hardware_config,
        mapper=mapper,
        bridge=bridge,
        hz=args.hz,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        slider_space=args.slider_space,
        viewer=viewer,
        disable_on_exit=args.disable_on_exit,
    )

    def _request_stop(_signum: int, _frame: Any) -> None:
        app.close()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
