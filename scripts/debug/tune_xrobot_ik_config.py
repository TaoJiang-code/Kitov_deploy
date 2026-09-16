#!/usr/bin/env python3
"""Tune local XRobot IK JSON parameters against one saved frame."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.gmr_online import DEFAULT_GMR_ROOT, OnlineGMRRetargeter, ROBOT_CONFIGS

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError as exc:  # pragma: no cover - depends on system Tk install.
    tk = None
    ttk = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune an XRobot IK config JSON on a saved frame.")
    parser.add_argument("frame_json", type=Path, help="Saved JSON produced by scripts/debug/xrobot_retarget.py.")
    parser.add_argument("--robot", choices=sorted(ROBOT_CONFIGS), default="openarm_v1", help="Target robot alias.")
    parser.add_argument("--ik-config", type=Path, default=None, help="Override robot IK config JSON.")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT, help="Deprecated compatibility option. Local GMR code is used.")
    parser.add_argument("--hz", type=float, default=50.0, help="Viewer refresh rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds. 0 means run until Ctrl-C.")
    parser.add_argument("--actual-human-height", type=float, default=None, help="Optional human height used to scale IK targets.")
    parser.add_argument("--solver", default="daqp", help="IK solver passed to mink/qpsolvers.")
    parser.add_argument("--damping", type=float, default=5e-1, help="IK damping passed to GMR.")
    parser.add_argument("--ground-offset", type=float, default=0.0, help="Subtract this z offset from all human targets.")
    parser.add_argument("--offset-to-ground", action="store_true", help="Shift the saved frame so the lowest foot target sits above ground.")
    parser.add_argument("--use-velocity-limit", action="store_true", help="Enable GMR velocity limits.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Repeated IK steps on the same frame before display.")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True, help="Open MuJoCo viewer.")
    parser.add_argument("--ik-editor", action=argparse.BooleanOptionalAction, default=True, help="Open JSON tuning panel.")
    parser.add_argument("--watch-ik-config", action=argparse.BooleanOptionalAction, default=True, help="Reload JSON when the file changes.")
    parser.add_argument("--reload-check-interval", type=float, default=0.25, help="Seconds between JSON mtime checks.")
    parser.add_argument("--show-human", action=argparse.BooleanOptionalAction, default=True, help="Draw human targets in the viewer.")
    parser.add_argument("--show-all-human", action=argparse.BooleanOptionalAction, default=True, help="Draw all converted XRobot body joints.")
    parser.add_argument("--human-axes-only", action=argparse.BooleanOptionalAction, default=True, help="Draw human target axes without blue spheres.")
    parser.add_argument("--show-human-name", action=argparse.BooleanOptionalAction, default=True, help="Draw human joint names.")
    parser.add_argument("--quiet-gmr", action="store_true", help="Suppress GMR model/body/dof listing.")
    return parser.parse_args()


def _as_pose_dict(raw: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            pos = value.get("position_rhs")
            quat = value.get("quat_wxyz_rhs")
        else:
            pos = value[0]
            quat = value[1]
        converted[str(name)] = (
            np.asarray(pos, dtype=np.float64).reshape(3),
            np.asarray(quat, dtype=np.float64).reshape(4),
        )
    return converted


def _load_body_frame(path: Path) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if "converted_body" not in payload:
        raise KeyError(f"{path} does not contain 'converted_body'")
    return payload, _as_pose_dict(payload["converted_body"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def _find_entry_for_human(table: dict[str, list[Any]], human_name: str) -> str | None:
    for robot_frame, entry in table.items():
        if entry and entry[0] == human_name:
            return robot_frame
    return None


def _config_human_names(config: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for entry in config.get(table_name, {}).values():
            if entry:
                names.add(str(entry[0]))
    return names


def _filter_human_data(
    human_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    names: set[str] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    if human_data is None or names is None:
        return human_data
    return {
        name: pose
        for name, pose in human_data.items()
        if name in names
    }


class IKJsonTuner:
    def __init__(
        self,
        *,
        config_path: Path,
        robot_body_names: list[str],
        human_body_names: list[str],
        on_saved: Callable[[], None],
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError(f"Tkinter is unavailable: {TK_IMPORT_ERROR}")
        self.config_path = config_path
        self.robot_body_names = robot_body_names
        self.human_body_names = human_body_names
        self.on_saved = on_saved
        self.closed = False
        self._updating = False
        self._human_scale_after_id: str | None = None

        self.config = self._load()
        self.has_target_origin = "target_origin" in self.config
        default_display_names = _config_human_names(self.config)
        self.root = tk.Tk()
        self.root.title(f"IK JSON Tuner - {self.config_path.name}")
        self.root.geometry("1220x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value=str(self.config_path))
        self.table_var = tk.StringVar(value="ik_match_table1")
        self.selected_robot_frame_var = tk.StringVar(value="")
        self.robot_frame_var = tk.StringVar(value="")
        self.human_body_var = tk.StringVar(value="")
        self.pos_weight_var = tk.StringVar(value="0.0")
        self.rot_weight_var = tk.StringVar(value="0.0")
        self.pos_offset_vars = [tk.StringVar(value="0.0") for _ in range(3)]
        self.quat_vars = [tk.StringVar(value=value) for value in ("1.0", "0.0", "0.0", "0.0")]
        self.rot_delta_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.rot_delta_scales: list[Any] = []
        self.rot_space_var = tk.StringVar(value="local")
        self.human_display_vars = {
            name: tk.BooleanVar(value=name in default_display_names)
            for name in self.human_body_names
        }
        self.human_scale_vars = {
            name: tk.DoubleVar(value=float(value))
            for name, value in self.config.get("human_scale_table", {}).items()
        }
        for var in self.human_scale_vars.values():
            var.trace_add("write", self._schedule_human_scale_apply)

        target_origin = self.config.get("target_origin", {})
        self.target_human_origin_var = tk.StringVar(value=str(target_origin.get("human_origin", "")))
        self.target_root_rotation_body_var = tk.StringVar(value=str(target_origin.get("root_rotation_body", target_origin.get("human_origin", ""))))
        self.target_lock_root_yaw_var = tk.BooleanVar(value=bool(target_origin.get("lock_root_yaw", False)))
        self.target_heading_yaw_offset_var = tk.DoubleVar(value=float(target_origin.get("heading_yaw_offset_deg", 0.0)))
        self.target_robot_origin_var = tk.StringVar(value=str(target_origin.get("robot_origin", "")))
        self.target_pos_offset_vars = [
            tk.StringVar(value=str(value)) for value in target_origin.get("position_offset", [0.0, 0.0, 0.0])
        ]
        self.target_quat_vars = [
            tk.StringVar(value=str(value)) for value in target_origin.get("rotation_offset", [1.0, 0.0, 0.0, 0.0])
        ]

        self._build_ui()
        self._refresh_entry_list()

    def _load(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _save(self, message: str) -> None:
        _write_json(self.config_path, self.config)
        self.status_var.set(f"[{time.strftime('%H:%M:%S')}] saved: {message}")
        self.on_saved()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Reload JSON", command=self.reload_from_disk).grid(row=0, column=1, padx=4)
        ttk.Button(header, text="Save Entry", command=self.save_selected_entry).grid(row=0, column=2, padx=4)
        if self.has_target_origin:
            ttk.Button(header, text="Save Target Origin", command=self.save_target_origin).grid(row=0, column=3, padx=4)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        ik_tab = ttk.Frame(notebook, padding=10)
        notebook.add(ik_tab, text="IK Match")
        self._build_ik_tab(ik_tab)

        if self.human_scale_vars:
            scale_tab = ttk.Frame(notebook, padding=10)
            notebook.add(scale_tab, text="Human Scale")
            self._build_human_scale_tab(scale_tab)

        if self.has_target_origin:
            target_tab = ttk.Frame(notebook, padding=10)
            notebook.add(target_tab, text="Target Origin")
            self._build_target_tab(target_tab)

        display_tab = ttk.Frame(notebook, padding=10)
        notebook.add(display_tab, text="Human Display")
        self._build_display_tab(display_tab)

    def _build_ik_tab(self, parent: Any) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(top, text="Table").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(top, textvariable=self.table_var, values=["ik_match_table1", "ik_match_table2"], state="readonly", width=18)
        combo.grid(row=0, column=1, padx=(8, 0), sticky="w")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_entry_list())

        left = ttk.Frame(parent)
        left.grid(row=1, column=0, sticky="nsw", pady=(10, 0))
        ttk.Label(left, text="robot frame -> human body").grid(row=0, column=0, sticky="w")
        self.entry_listbox = tk.Listbox(left, height=30, width=42, exportselection=False)
        self.entry_listbox.grid(row=1, column=0, sticky="nsw", pady=(6, 0))
        self.entry_listbox.bind("<<ListboxSelect>>", lambda _event: self._populate_selected_entry())

        form = ttk.Frame(parent, padding=(24, 10, 0, 0))
        form.grid(row=1, column=1, sticky="nsew")
        form.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(form, text="Robot frame").grid(row=row, column=0, sticky="w")
        ttk.Combobox(form, textvariable=self.robot_frame_var, values=self.robot_body_names, width=34).grid(row=row, column=1, sticky="w")
        ttk.Button(form, text="Sync", command=lambda: self.sync_field_to_other_table("robot_frame")).grid(row=row, column=2, sticky="w", padx=(12, 0))

        row += 1
        ttk.Label(form, text="Human body").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(form, textvariable=self.human_body_var, values=self.human_body_names, width=34).grid(row=row, column=1, sticky="w", pady=(10, 0))
        ttk.Button(form, text="Sync", command=lambda: self.sync_field_to_other_table("human_body")).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(10, 0))

        row += 1
        ttk.Label(form, text="Position weight").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.pos_weight_var, width=16).grid(row=row, column=1, sticky="w", pady=(10, 0))
        ttk.Button(form, text="Sync", command=lambda: self.sync_field_to_other_table("pos_weight")).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(10, 0))

        row += 1
        ttk.Label(form, text="Rotation weight").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.rot_weight_var, width=16).grid(row=row, column=1, sticky="w", pady=(10, 0))
        ttk.Button(form, text="Sync", command=lambda: self.sync_field_to_other_table("rot_weight")).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(10, 0))

        row += 1
        ttk.Label(form, text="Position offset [x y z]").grid(row=row, column=0, sticky="w", pady=(10, 0))
        pos_frame = ttk.Frame(form)
        pos_frame.grid(row=row, column=1, sticky="w", pady=(10, 0))
        for idx, axis in enumerate(("x", "y", "z")):
            ttk.Label(pos_frame, text=axis).grid(row=0, column=idx * 2, sticky="w")
            ttk.Entry(pos_frame, textvariable=self.pos_offset_vars[idx], width=10).grid(row=0, column=idx * 2 + 1, padx=(2, 10))
        ttk.Button(form, text="Sync", command=lambda: self.sync_field_to_other_table("pos_offset")).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(10, 0))

        row += 1
        ttk.Label(form, text="Rotation quat [w x y z]").grid(row=row, column=0, sticky="w", pady=(10, 0))
        quat_frame = ttk.Frame(form)
        quat_frame.grid(row=row, column=1, sticky="w", pady=(10, 0))
        for idx, name in enumerate(("w", "x", "y", "z")):
            ttk.Label(quat_frame, text=name).grid(row=0, column=idx * 2, sticky="w")
            ttk.Entry(quat_frame, textvariable=self.quat_vars[idx], width=10).grid(row=0, column=idx * 2 + 1, padx=(2, 10))
        ttk.Button(form, text="Sync", command=lambda: self.sync_field_to_other_table("rot_offset")).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(10, 0))

        row += 1
        ttk.Label(form, text="Rotation delta xyz deg").grid(row=row, column=0, sticky="nw", pady=(10, 0))
        delta_frame = ttk.Frame(form)
        delta_frame.grid(row=row, column=1, sticky="ew", pady=(10, 0))
        delta_frame.columnconfigure(2, weight=1)
        self.rot_delta_scales = []
        for idx, axis in enumerate(("x", "y", "z")):
            ttk.Label(delta_frame, text=axis).grid(row=idx, column=0, sticky="w", pady=2)
            ttk.Entry(delta_frame, textvariable=self.rot_delta_vars[idx], width=9).grid(row=idx, column=1, sticky="w", padx=(4, 8), pady=2)
            scale = tk.Scale(
                delta_frame,
                from_=-180,
                to=180,
                orient=tk.HORIZONTAL,
                resolution=1.0,
                length=420,
                variable=self.rot_delta_vars[idx],
            )
            scale.grid(row=idx, column=2, sticky="ew", pady=2)
            self.rot_delta_scales.append(scale)
        delta_buttons = ttk.Frame(form)
        delta_buttons.grid(row=row, column=2, sticky="nw", padx=(12, 0), pady=(10, 0))
        ttk.Combobox(delta_buttons, textvariable=self.rot_space_var, values=["local", "global"], state="readonly", width=8).grid(row=0, column=0, sticky="w")
        ttk.Button(delta_buttons, text="Apply Delta", command=self.apply_rotation_delta).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(delta_buttons, text="Reset Delta", command=self.reset_rotation_delta).grid(row=2, column=0, sticky="w", pady=(8, 0))

        row += 1
        buttons = ttk.Frame(form)
        buttons.grid(row=row, column=0, columnspan=2, sticky="w", pady=(18, 0))
        ttk.Button(buttons, text="Save Entry", command=self.save_selected_entry).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Sync Whole Entry", command=lambda: self.sync_field_to_other_table("all")).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Reload JSON", command=self.reload_from_disk).grid(row=0, column=2)

    def _build_target_tab(self, parent: Any) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="human_origin").grid(row=0, column=0, sticky="w")
        ttk.Combobox(parent, textvariable=self.target_human_origin_var, values=self.human_body_names, width=34).grid(row=0, column=1, sticky="w")
        ttk.Label(parent, text="root_rotation_body").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(parent, textvariable=self.target_root_rotation_body_var, values=self.human_body_names, width=34).grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Checkbutton(parent, text="lock root yaw", variable=self.target_lock_root_yaw_var).grid(row=2, column=1, sticky="w", pady=(10, 0))
        ttk.Label(parent, text="heading_yaw_offset_deg").grid(row=3, column=0, sticky="w", pady=(10, 0))
        yaw_frame = ttk.Frame(parent)
        yaw_frame.grid(row=3, column=1, sticky="ew", pady=(10, 0))
        yaw_frame.columnconfigure(1, weight=1)
        ttk.Entry(yaw_frame, textvariable=self.target_heading_yaw_offset_var, width=10).grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Scale(
            yaw_frame,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            resolution=1.0,
            length=420,
            variable=self.target_heading_yaw_offset_var,
        ).grid(row=0, column=1, sticky="ew")
        ttk.Label(parent, text="robot_origin").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(parent, textvariable=self.target_robot_origin_var, values=self.robot_body_names, width=34).grid(row=4, column=1, sticky="w", pady=(10, 0))

        ttk.Label(parent, text="position_offset [x y z]").grid(row=5, column=0, sticky="w", pady=(10, 0))
        pos_frame = ttk.Frame(parent)
        pos_frame.grid(row=5, column=1, sticky="w", pady=(10, 0))
        for idx, axis in enumerate(("x", "y", "z")):
            ttk.Label(pos_frame, text=axis).grid(row=0, column=idx * 2, sticky="w")
            ttk.Entry(pos_frame, textvariable=self.target_pos_offset_vars[idx], width=10).grid(row=0, column=idx * 2 + 1, padx=(2, 10))

        ttk.Label(parent, text="rotation_offset quat [w x y z]").grid(row=6, column=0, sticky="w", pady=(10, 0))
        quat_frame = ttk.Frame(parent)
        quat_frame.grid(row=6, column=1, sticky="w", pady=(10, 0))
        for idx, axis in enumerate(("w", "x", "y", "z")):
            ttk.Label(quat_frame, text=axis).grid(row=0, column=idx * 2, sticky="w")
            ttk.Entry(quat_frame, textvariable=self.target_quat_vars[idx], width=10).grid(row=0, column=idx * 2 + 1, padx=(2, 10))

        ttk.Button(parent, text="Save Target Origin", command=self.save_target_origin).grid(row=7, column=0, columnspan=2, sticky="w", pady=(18, 0))

    def _build_human_scale_tab(self, parent: Any) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        buttons = ttk.Frame(parent)
        buttons.grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Save Human Scale", command=self.save_human_scale_table).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Reset From JSON", command=self.reload_human_scale_vars).grid(row=0, column=1, padx=(0, 8))

        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        frame.columnconfigure(2, weight=1)

        for row, name in enumerate(self.human_scale_vars):
            ttk.Label(frame, text=name, width=22).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frame, textvariable=self.human_scale_vars[name], width=10).grid(row=row, column=1, sticky="w", padx=(8, 10), pady=4)
            tk.Scale(
                frame,
                from_=0.0,
                to=2.0,
                orient=tk.HORIZONTAL,
                resolution=0.01,
                length=520,
                variable=self.human_scale_vars[name],
            ).grid(row=row, column=2, sticky="ew", pady=4)

    def _build_display_tab(self, parent: Any) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        buttons = ttk.Frame(parent)
        buttons.grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Show Match Table", command=self.select_match_table_human_display).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Show All", command=self.select_all_human_display).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Hide All", command=self.clear_human_display).grid(row=0, column=2, padx=(0, 8))

        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(12, 0))

        columns = 4
        for idx, name in enumerate(self.human_body_names):
            check = ttk.Checkbutton(
                frame,
                text=name,
                variable=self.human_display_vars[name],
                command=self._update_display_status,
            )
            check.grid(row=idx // columns, column=idx % columns, sticky="w", padx=(0, 28), pady=4)
        self._update_display_status()

    def _table(self, table_name: str | None = None) -> dict[str, list[Any]]:
        return self.config[table_name or self.table_var.get()]

    def _refresh_entry_list(self) -> None:
        selected = self.selected_robot_frame_var.get()
        self.entry_listbox.delete(0, tk.END)
        frames = list(self._table().keys())
        for robot_frame in frames:
            entry = self._table()[robot_frame]
            self.entry_listbox.insert(tk.END, f"{robot_frame} -> {entry[0]}")
        if not frames:
            return
        if selected not in frames:
            selected = frames[0]
        self.entry_listbox.selection_clear(0, tk.END)
        self.entry_listbox.selection_set(frames.index(selected))
        self._populate_selected_entry()

    def _populate_selected_entry(self) -> None:
        selection = self.entry_listbox.curselection()
        if not selection:
            return
        robot_frame = list(self._table().keys())[selection[0]]
        entry = self._table()[robot_frame]
        self.selected_robot_frame_var.set(robot_frame)
        self.robot_frame_var.set(robot_frame)
        self.human_body_var.set(str(entry[0]))
        self.pos_weight_var.set(f"{float(entry[1]):.6f}")
        self.rot_weight_var.set(f"{float(entry[2]):.6f}")
        for idx, value in enumerate(entry[3]):
            self.pos_offset_vars[idx].set(f"{float(value):.6f}")
        for idx, value in enumerate(entry[4]):
            self.quat_vars[idx].set(f"{float(value):.8f}")
        self.reset_rotation_delta()

    def save_selected_entry(self) -> None:
        old_robot_frame = self.selected_robot_frame_var.get()
        new_robot_frame = self.robot_frame_var.get().strip()
        if not old_robot_frame or not new_robot_frame:
            self.status_var.set("Select an IK entry first")
            return
        entry = [
            self.human_body_var.get().strip(),
            float(self.pos_weight_var.get()),
            float(self.rot_weight_var.get()),
            [float(var.get()) for var in self.pos_offset_vars],
            [float(var.get()) for var in self.quat_vars],
        ]
        table = self._table()
        if old_robot_frame != new_robot_frame:
            table.pop(old_robot_frame, None)
        table[new_robot_frame] = entry
        self.selected_robot_frame_var.set(new_robot_frame)
        self._save(f"{self.table_var.get()} {new_robot_frame}")
        self._refresh_entry_list()

    def sync_field_to_other_table(self, field_name: str) -> None:
        old_robot_frame = self.selected_robot_frame_var.get()
        new_robot_frame = self.robot_frame_var.get().strip()
        if not old_robot_frame or not new_robot_frame:
            self.status_var.set("Select an IK entry first")
            return

        self.save_selected_entry()
        source_table = self.table_var.get()
        other_table = "ik_match_table2" if source_table == "ik_match_table1" else "ik_match_table1"
        source_entry = list(self.config[source_table][new_robot_frame])

        target_frame = new_robot_frame
        target_entry = self.config[other_table].get(target_frame)
        if target_entry is None:
            old_other_frame = _find_entry_for_human(self.config[other_table], source_entry[0])
            if old_other_frame is not None:
                target_frame = old_other_frame
                target_entry = self.config[other_table][old_other_frame]
            else:
                target_entry = list(source_entry)

        if field_name == "all":
            target_entry = list(source_entry)
        elif field_name == "robot_frame":
            pass
        elif field_name == "human_body":
            target_entry[0] = source_entry[0]
        elif field_name == "pos_weight":
            target_entry[1] = source_entry[1]
        elif field_name == "rot_weight":
            target_entry[2] = source_entry[2]
        elif field_name == "pos_offset":
            target_entry[3] = list(source_entry[3])
        elif field_name == "rot_offset":
            target_entry[4] = list(source_entry[4])
        else:
            raise ValueError(f"Unknown sync field: {field_name}")

        if target_frame != new_robot_frame:
            self.config[other_table].pop(target_frame, None)
        self.config[other_table][new_robot_frame] = target_entry
        self._save(f"synced {field_name} to {other_table}:{new_robot_frame}")

    def apply_rotation_delta(self) -> None:
        quat = [float(var.get()) for var in self.quat_vars]
        delta = [float(var.get()) for var in self.rot_delta_vars]
        current = Rotation.from_quat(quat, scalar_first=True)
        delta_rot = Rotation.from_euler("xyz", delta, degrees=True)
        updated = current * delta_rot if self.rot_space_var.get() == "local" else delta_rot * current
        for idx, value in enumerate(updated.as_quat(scalar_first=True)):
            self.quat_vars[idx].set(f"{float(value):.8f}")
        self.reset_rotation_delta()
        self.save_selected_entry()

    def reset_rotation_delta(self) -> None:
        for var in self.rot_delta_vars:
            var.set(0.0)

    def save_human_scale_table(self) -> None:
        self._save_human_scale_table("human_scale_table")

    def _schedule_human_scale_apply(self, *_args: Any) -> None:
        if self._updating:
            return
        if self._human_scale_after_id is not None:
            self.root.after_cancel(self._human_scale_after_id)
        self._human_scale_after_id = self.root.after(150, self._auto_apply_human_scale_table)

    def _auto_apply_human_scale_table(self) -> None:
        self._human_scale_after_id = None
        self._save_human_scale_table("human_scale_table auto")

    def _save_human_scale_table(self, message: str) -> None:
        if "human_scale_table" not in self.config:
            self.status_var.set("Current JSON has no human_scale_table")
            return
        try:
            scale_table = {
                name: float(var.get())
                for name, var in self.human_scale_vars.items()
            }
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(f"Invalid human scale value: {exc}")
            return
        self.config["human_scale_table"] = scale_table
        self._save(message)

    def reload_human_scale_vars(self) -> None:
        scale_table = self.config.get("human_scale_table", {})
        self._updating = True
        try:
            for name, var in self.human_scale_vars.items():
                if name in scale_table:
                    var.set(float(scale_table[name]))
        finally:
            self._updating = False
        self.status_var.set(f"[{time.strftime('%H:%M:%S')}] reloaded human_scale_table")

    def save_target_origin(self) -> None:
        if not self.has_target_origin:
            self.status_var.set("Current JSON has no target_origin")
            return
        self.config["target_origin"] = {
            "human_origin": self.target_human_origin_var.get().strip(),
            "root_rotation_body": self.target_root_rotation_body_var.get().strip(),
            "lock_root_yaw": bool(self.target_lock_root_yaw_var.get()),
            "heading_yaw_offset_deg": float(self.target_heading_yaw_offset_var.get()),
            "robot_origin": self.target_robot_origin_var.get().strip(),
            "position_offset": [float(var.get()) for var in self.target_pos_offset_vars],
            "rotation_offset": [float(var.get()) for var in self.target_quat_vars],
        }
        self._save("target_origin")

    def reload_from_disk(self) -> None:
        self.config = self._load()
        self.reload_human_scale_vars()
        if self.has_target_origin:
            target_origin = self.config.get("target_origin", {})
            self.target_human_origin_var.set(str(target_origin.get("human_origin", "")))
            self.target_root_rotation_body_var.set(str(target_origin.get("root_rotation_body", target_origin.get("human_origin", ""))))
            self.target_lock_root_yaw_var.set(bool(target_origin.get("lock_root_yaw", False)))
            self.target_heading_yaw_offset_var.set(float(target_origin.get("heading_yaw_offset_deg", 0.0)))
            self.target_robot_origin_var.set(str(target_origin.get("robot_origin", "")))
            for idx, value in enumerate(target_origin.get("position_offset", [0.0, 0.0, 0.0])):
                self.target_pos_offset_vars[idx].set(str(value))
            for idx, value in enumerate(target_origin.get("rotation_offset", [1.0, 0.0, 0.0, 0.0])):
                self.target_quat_vars[idx].set(str(value))
        self._refresh_entry_list()
        self.status_var.set(f"[{time.strftime('%H:%M:%S')}] reloaded {self.config_path}")

    def selected_human_display_names(self) -> set[str]:
        return {
            name
            for name, var in self.human_display_vars.items()
            if bool(var.get())
        }

    def selected_table_name(self) -> str:
        return self.table_var.get()

    def _set_human_display_names(self, names: set[str]) -> None:
        for name, var in self.human_display_vars.items():
            var.set(name in names)
        self._update_display_status()

    def select_match_table_human_display(self) -> None:
        self._set_human_display_names(_config_human_names(self.config))

    def select_all_human_display(self) -> None:
        self._set_human_display_names(set(self.human_body_names))

    def clear_human_display(self) -> None:
        self._set_human_display_names(set())

    def _update_display_status(self) -> None:
        selected = len(self.selected_human_display_names())
        self.status_var.set(f"[{time.strftime('%H:%M:%S')}] displaying {selected}/{len(self.human_body_names)} human frames")

    def process_events(self) -> bool:
        if self.closed:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            return False
        return True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> int:
    args = _parse_args()
    payload, body = _load_body_frame(args.frame_json)
    stop = False
    reload_requested = True

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    def _request_reload() -> None:
        nonlocal reload_requested
        reload_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def build_retargeter() -> OnlineGMRRetargeter:
        retargeter = OnlineGMRRetargeter(
            args.robot,
            gmr_root=args.gmr_root,
            actual_human_height=args.actual_human_height,
            solver=args.solver,
            damping=args.damping,
            use_velocity_limit=args.use_velocity_limit,
            ik_config_path=args.ik_config,
            verbose=not args.quiet_gmr,
        )
        retargeter.set_ground_offset(args.ground_offset)
        return retargeter

    retargeter = build_retargeter()
    ik_config_path = (args.ik_config or retargeter.config.ik_config_path).expanduser()
    last_mtime = ik_config_path.stat().st_mtime
    viewer = retargeter.make_viewer(motion_fps=args.hz) if args.viewer else None
    editor = None
    if args.ik_editor:
        editor = IKJsonTuner(
            config_path=ik_config_path,
            robot_body_names=sorted(retargeter.robot_body_names),
            human_body_names=sorted(body),
            on_saved=_request_reload,
        )

    qpos = None
    print(
        "[tune_xrobot_ik_config] started "
        f"input={args.frame_json} saved_robot={payload.get('robot')} target_robot={args.robot} "
        f"ik_config={ik_config_path}"
    )
    print("Edit and save the JSON panel to refresh the current pose. Ctrl-C exits.")

    start_time = time.monotonic()
    try:
        while not stop:
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break
            if editor is not None and not editor.process_events():
                editor = None

            if args.watch_ik_config:
                current_mtime = ik_config_path.stat().st_mtime
                if current_mtime > last_mtime:
                    reload_requested = True
                    last_mtime = current_mtime

            if reload_requested:
                try:
                    retargeter = build_retargeter()
                    qpos = None
                    for _ in range(max(args.warmup_steps, 1)):
                        qpos = retargeter.retarget(body, offset_to_ground=args.offset_to_ground)
                    assert qpos is not None
                    print(
                        "[tune_xrobot_ik_config] reloaded "
                        f"qpos={len(qpos)} head={[round(float(v), 4) for v in qpos[: min(8, len(qpos))]]}"
                    )
                except Exception as exc:
                    print(f"[tune_xrobot_ik_config] reload failed: {exc}")
                reload_requested = False

            if viewer is not None and qpos is not None:
                visible_names = editor.selected_human_display_names() if editor is not None else None
                selected_table = editor.selected_table_name() if editor is not None else "ik_match_table1"
                debug_human_all = (
                    retargeter.prepare_debug_human_data(
                        body,
                        offset_to_ground=args.offset_to_ground,
                        include_unscaled=args.show_all_human,
                        task_table=selected_table,
                    )
                    if args.show_human
                    else None
                )
                debug_human = _filter_human_data(debug_human_all, visible_names)
                viewer.step_qpos(
                    qpos,
                    human_motion_data=debug_human,
                    show_human_body_name=args.show_human_name,
                    show_human_points=not args.human_axes_only,
                    rate_limit=True,
                )
            else:
                time.sleep(min(1.0 / max(args.hz, 1e-6), 0.05))
    finally:
        if editor is not None:
            editor.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
