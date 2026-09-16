"""MuJoCo sim2sim bridge for Kitov policy q_target output."""

from __future__ import annotations

import tempfile
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from kitov_deploy.mjcf_utils import prepared_mjcf_path
from kitov_deploy.policy_runtime import RobotPolicyConfig, RobotState


def _load_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import mujoco. Install it in the Kitov_deploy conda environment first:\n"
            "python -m pip install mujoco"
        ) from exc
    return mujoco


@dataclass(frozen=True)
class MujocoSimStatus:
    time: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    torque: np.ndarray


def _format_floor_rgb(rgb: tuple[float, float, float]) -> str:
    return " ".join(f"{float(channel):.3f}" for channel in rgb)


def _scale_floor_rgb(rgb: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(min(max(float(channel) * scale, 0.0), 1.0) for channel in rgb)


def _has_named_xml_element(xml_text: str, tag: str, name: str) -> bool:
    start = 0
    open_token = f"<{tag}"
    while True:
        tag_start = xml_text.find(open_token, start)
        if tag_start < 0:
            return False
        tag_end = xml_text.find(">", tag_start)
        if tag_end < 0:
            return False
        tag_text = xml_text[tag_start:tag_end]
        if f'name="{name}"' in tag_text or f"name='{name}'" in tag_text:
            return True
        start = tag_end + 1


def inject_floor_scene_xml(
    xml_text: str,
    ground_rgb: tuple[float, float, float] = (0.35, 0.35, 0.35),
) -> str:
    if "<asset>" not in xml_text or "</asset>" not in xml_text:
        raise ValueError("MJCF XML structure error: expected <asset>...</asset> block for floor assets.")
    if "</worldbody>" not in xml_text:
        raise ValueError("MJCF XML structure error: expected </worldbody> block for floor geom.")

    if "<visual>" not in xml_text:
        visual_xml = """\
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.18 0.18 0.18" specular="0.8 0.8 0.8"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
"""
        asset_open = xml_text.find("<asset>")
        xml_text = xml_text[:asset_open] + visual_xml + xml_text[asset_open:]

    ground_rgb_dark = _scale_floor_rgb(ground_rgb, 0.75)
    asset_parts: list[str] = []
    if not _has_named_xml_element(xml_text, "texture", "groundplane"):
        asset_parts.append(
            f"""\
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="{_format_floor_rgb(ground_rgb)}" rgb2="{_format_floor_rgb(ground_rgb_dark)}" markrgb="0.8 0.8 0.8" width="300" height="300"/>
"""
        )
    if not _has_named_xml_element(xml_text, "material", "groundplane"):
        asset_parts.append(
            """\
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""
        )
    if asset_parts:
        asset_close = xml_text.find("</asset>")
        xml_text = xml_text[:asset_close] + "".join(asset_parts) + xml_text[asset_close:]

    if _has_named_xml_element(xml_text, "geom", "floor"):
        return xml_text

    worldbody_xml = """\
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""
    worldbody_close = xml_text.find("</worldbody>")
    return xml_text[:worldbody_close] + worldbody_xml + xml_text[worldbody_close:]


def mjcf_has_floor(mjcf_path: str | Path) -> bool:
    xml_text = Path(mjcf_path).expanduser().read_text(encoding="utf-8")
    return _has_named_xml_element(xml_text, "geom", "floor")


def _absolutize_compiler_path_attrs(xml_text: str, source_dir: Path) -> str:
    def _absolute_attr(attr_name: str, tag_text: str) -> str:
        attr_re = re.compile(rf'({attr_name}\s*=\s*["\'])([^"\']+)(["\'])')
        match = attr_re.search(tag_text)
        if match is None:
            suffix = "/>" if tag_text.endswith("/>") else ">"
            prefix = tag_text[: -len(suffix)]
            return prefix + f' {attr_name}="{source_dir.as_posix()}"' + suffix
        value = Path(match.group(2)).expanduser()
        if not value.is_absolute():
            value = (source_dir / value).resolve(strict=False)
        return attr_re.sub(rf'\1{value.as_posix()}\3', tag_text, count=1)

    compiler_re = re.compile(r"<compiler\b[^>]*>")
    match = compiler_re.search(xml_text)
    if match is None:
        if "<include" in xml_text:
            return xml_text
        mujoco_open_end = xml_text.find(">")
        if mujoco_open_end < 0:
            return xml_text
        compiler = f'\n  <compiler meshdir="{source_dir.as_posix()}"/>'
        return xml_text[: mujoco_open_end + 1] + compiler + xml_text[mujoco_open_end + 1 :]

    compiler_tag = match.group(0)
    compiler_tag = _absolute_attr("meshdir", compiler_tag)
    return xml_text[: match.start()] + compiler_tag + xml_text[match.end() :]


def _absolutize_include_file_attrs(xml_text: str, source_dir: Path) -> str:
    include_re = re.compile(r"(<include\b[^>]*\bfile\s*=\s*[\"'])([^\"']+)([\"'][^>]*>)")

    def _replace(match: re.Match[str]) -> str:
        value = Path(match.group(2)).expanduser()
        if not value.is_absolute():
            value = (source_dir / value).resolve(strict=False)
        return f"{match.group(1)}{value.as_posix()}{match.group(3)}"

    return include_re.sub(_replace, xml_text)


@contextmanager
def temp_mjcf_with_floor(mjcf_path: str | Path) -> Iterator[Path]:
    source_path = Path(mjcf_path).expanduser()
    if not source_path.is_absolute():
        source_path = source_path.resolve(strict=False)

    xml_text = source_path.read_text(encoding="utf-8")
    xml_text = _absolutize_include_file_attrs(xml_text, source_path.parent)
    xml_text = _absolutize_compiler_path_attrs(xml_text, source_path.parent)
    viewer_xml_text = inject_floor_scene_xml(xml_text)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".kitov_viewer_floor_",
            dir="/tmp",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write(viewer_xml_text)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        yield temp_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


class MujocoPolicySim:
    """Apply policy q_target to a MuJoCo model using PD torque control."""

    def __init__(
        self,
        robot_config: RobotPolicyConfig,
        *,
        hz: float = 50.0,
        sim_substeps: int | None = None,
        viewer: bool = False,
        add_floor: bool = True,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        elastic_band: bool = False,
        elastic_body_name: str | None = None,
        elastic_stiffness: float = 200.0,
        elastic_damping: float = 100.0,
        elastic_length: float = 0.0,
        elastic_point: tuple[float, float, float] = (0.0, 0.0, 3.0),
        start_policy_enabled: bool = True,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        self.mujoco = _load_mujoco()
        self.robot_config = robot_config
        if not robot_config.xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {robot_config.xml_path}")

        self._prepared_context = prepared_mjcf_path(robot_config.xml_path)
        prepared_xml_path = self._prepared_context.__enter__()
        self._floor_context = temp_mjcf_with_floor(prepared_xml_path) if add_floor and not mjcf_has_floor(prepared_xml_path) else None
        try:
            xml_path = self._floor_context.__enter__() if self._floor_context is not None else prepared_xml_path
            self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        except Exception:
            if self._floor_context is not None:
                self._floor_context.__exit__(None, None, None)
                self._floor_context = None
            self._prepared_context.__exit__(None, None, None)
            raise
        self.data = self.mujoco.MjData(self.model)
        if robot_config.sim_timestep is not None:
            self.model.opt.timestep = float(robot_config.sim_timestep)

        if self.model.nq != 7 + robot_config.num_dof:
            raise ValueError(
                f"Robot XML nq={self.model.nq} does not match config num_dof={robot_config.num_dof}"
            )
        if self.model.nu < robot_config.num_dof:
            raise ValueError(
                f"Robot XML nu={self.model.nu} is smaller than config num_dof={robot_config.num_dof}"
            )

        self.qpos_adrs: list[int] = []
        self.qvel_adrs: list[int] = []
        self.actuator_ids: list[int] = []
        for name in robot_config.control_joint_names:
            joint = self.model.joint(name)
            self.qpos_adrs.append(int(self.model.jnt_qposadr[joint.id]))
            self.qvel_adrs.append(int(self.model.jnt_dofadr[joint.id]))
            self.actuator_ids.append(self._actuator_id_for_joint(name, joint.id))

        self.qpos_adrs_np = np.asarray(self.qpos_adrs, dtype=np.int32)
        self.qvel_adrs_np = np.asarray(self.qvel_adrs, dtype=np.int32)
        self.actuator_ids_np = np.asarray(self.actuator_ids, dtype=np.int32)
        self.model.dof_armature[self.qvel_adrs_np] = np.asarray(
            robot_config.sim_joint_armature,
            dtype=np.float64,
        )
        self.model.dof_frictionloss[self.qvel_adrs_np] = np.asarray(
            robot_config.sim_joint_frictionloss,
            dtype=np.float64,
        )
        self.kp = np.asarray(robot_config.sim_joint_kp, dtype=np.float64) * float(kp_scale)
        self.kd = np.asarray(robot_config.sim_joint_kd, dtype=np.float64) * float(kd_scale)
        self.effort_limit = np.asarray(robot_config.sim_effort_limit, dtype=np.float64)
        self.last_torque = np.zeros(robot_config.num_dof, dtype=np.float64)
        self.elastic_band = bool(elastic_band)
        self.elastic_stiffness = float(elastic_stiffness)
        self.elastic_damping = float(elastic_damping)
        self.elastic_length = float(elastic_length)
        self.elastic_point = np.asarray(elastic_point, dtype=np.float64).reshape(3)
        self.policy_enabled = bool(start_policy_enabled)
        self._external_key_callback = key_callback
        self.elastic_body_id: int | None = None
        if self.elastic_band:
            body_name = elastic_body_name or (
                "torso_link" if robot_config.robot_name == "g1" else robot_config.camera_body_name
            )
            if not body_name:
                body_name = robot_config.body_names[0]
            try:
                self.elastic_body_id = int(self.model.body(body_name).id)
            except KeyError as exc:
                raise KeyError(f"Elastic-band body {body_name!r} not found in {robot_config.xml_path}") from exc

        free_joint_ids = np.where(self.model.jnt_type == self.mujoco.mjtJoint.mjJNT_FREE)[0]
        if free_joint_ids.size == 0:
            raise ValueError(f"Robot XML has no free joint: {robot_config.xml_path}")
        self.root_qpos_adr = int(self.model.jnt_qposadr[int(free_joint_ids[0])])
        self.root_qvel_adr = int(self.model.jnt_dofadr[int(free_joint_ids[0])])

        period_s = 1.0 / max(float(hz), 1e-6)
        if sim_substeps is None or int(sim_substeps) <= 0:
            self.sim_substeps = max(1, int(round(period_s / float(self.model.opt.timestep))))
        else:
            self.sim_substeps = int(sim_substeps)

        self.viewer = None
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self._on_viewer_key,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._configure_camera()

        self.reset()

    def _actuator_id_for_joint(self, name: str, joint_id: int) -> int:
        try:
            return int(self.model.actuator(name).id)
        except KeyError:
            pass
        for actuator_id in range(self.model.nu):
            if int(self.model.actuator_trnid[actuator_id, 0]) == int(joint_id):
                return actuator_id
        raise KeyError(f"No actuator found for joint {name!r} in {self.robot_config.xml_path}")

    def _configure_camera(self) -> None:
        if self.viewer is None:
            return
        body_name = self.robot_config.camera_body_name or self.robot_config.body_names[0]
        try:
            body_id = self.model.body(body_name).id
        except KeyError:
            return
        self.viewer.cam.type = self.mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = body_id
        self.viewer.cam.distance = 3.0

    def _on_viewer_key(self, key: int) -> None:
        if int(key) in (ord("P"), ord("p")):
            self.set_damping_control()
        elif int(key) == ord("1"):
            self.set_policy_control()
        elif int(key) == ord("0"):
            self.reset_joints_to_zero()
        if self._external_key_callback is not None:
            self._external_key_callback(key)

    def is_policy_enabled(self) -> bool:
        return self.policy_enabled

    def toggle_policy_control(self) -> bool:
        self.policy_enabled = not self.policy_enabled
        mode = "policy" if self.policy_enabled else "damping"
        print(f"[mujoco_policy_sim] control mode: {mode}")
        return self.policy_enabled

    def set_policy_control(self) -> None:
        if not self.policy_enabled:
            print("[mujoco_policy_sim] control mode: policy")
        self.policy_enabled = True

    def set_damping_control(self) -> None:
        if self.policy_enabled:
            print("[mujoco_policy_sim] control mode: damping")
        self.policy_enabled = False

    def reset_joints_to_zero(self) -> None:
        state = self.robot_state()
        self.reset(
            RobotState(
                root_pos=state.root_pos,
                root_quat_wxyz=state.root_quat_wxyz,
                dof_pos=np.zeros(self.robot_config.num_dof, dtype=np.float32),
                root_lin_vel=np.zeros(3, dtype=np.float32),
                root_ang_vel=np.zeros(3, dtype=np.float32),
                dof_vel=np.zeros(self.robot_config.num_dof, dtype=np.float32),
            )
        )
        self.set_damping_control()
        print("[mujoco_policy_sim] reset joints to zero; control mode: damping")

    def reset(self, state: RobotState | None = None) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.last_torque[:] = 0.0
        if state is None:
            root_pos = self.robot_config.default_root_pos
            root_quat = self.robot_config.default_root_quat_wxyz
            dof_pos = self.robot_config.default_joint_angles
            root_lin_vel = np.zeros(3, dtype=np.float64)
            root_ang_vel = np.zeros(3, dtype=np.float64)
            dof_vel = np.zeros(self.robot_config.num_dof, dtype=np.float64)
        else:
            root_pos = np.asarray(state.root_pos, dtype=np.float64).reshape(3)
            root_quat = np.asarray(state.root_quat_wxyz, dtype=np.float64).reshape(4)
            dof_pos = np.asarray(state.dof_pos, dtype=np.float64).reshape(self.robot_config.num_dof)
            root_lin_vel = (
                np.zeros(3, dtype=np.float64)
                if state.root_lin_vel is None
                else np.asarray(state.root_lin_vel, dtype=np.float64).reshape(3)
            )
            root_ang_vel = (
                np.zeros(3, dtype=np.float64)
                if state.root_ang_vel is None
                else np.asarray(state.root_ang_vel, dtype=np.float64).reshape(3)
            )
            dof_vel = (
                np.zeros(self.robot_config.num_dof, dtype=np.float64)
                if state.dof_vel is None
                else np.asarray(state.dof_vel, dtype=np.float64).reshape(self.robot_config.num_dof)
            )

        norm = max(float(np.linalg.norm(root_quat)), 1e-8)
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = root_pos
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = root_quat / norm
        self.data.qpos[self.qpos_adrs_np] = dof_pos
        self.data.qvel[self.root_qvel_adr : self.root_qvel_adr + 3] = root_lin_vel
        self.data.qvel[self.root_qvel_adr + 3 : self.root_qvel_adr + 6] = root_ang_vel
        self.data.qvel[self.qvel_adrs_np] = dof_vel
        self.mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def robot_state(self) -> RobotState:
        return RobotState(
            root_pos=self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3].astype(np.float32).copy(),
            root_quat_wxyz=self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7].astype(np.float32).copy(),
            dof_pos=self.data.qpos[self.qpos_adrs_np].astype(np.float32).copy(),
            root_lin_vel=self.data.qvel[self.root_qvel_adr : self.root_qvel_adr + 3].astype(np.float32).copy(),
            root_ang_vel=self.data.qvel[self.root_qvel_adr + 3 : self.root_qvel_adr + 6].astype(np.float32).copy(),
            dof_vel=self.data.qvel[self.qvel_adrs_np].astype(np.float32).copy(),
        )

    def apply_q_target(self, q_target: np.ndarray) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=np.float64).reshape(self.robot_config.num_dof)
        q = self.data.qpos[self.qpos_adrs_np]
        dq = self.data.qvel[self.qvel_adrs_np]
        torque = self.kp * (q_target - q) - self.kd * dq
        torque = np.clip(torque, -self.effort_limit, self.effort_limit)

        limited = np.asarray(self.model.actuator_ctrllimited[self.actuator_ids_np], dtype=bool)
        if np.any(limited):
            ctrlrange = self.model.actuator_ctrlrange[self.actuator_ids_np]
            torque[limited] = np.clip(torque[limited], ctrlrange[limited, 0], ctrlrange[limited, 1])

        self.data.ctrl[self.actuator_ids_np] = torque
        self.last_torque = torque.copy()
        return self.last_torque

    def apply_damping(self) -> np.ndarray:
        dq = self.data.qvel[self.qvel_adrs_np]
        torque = -self.kd * dq
        torque = np.clip(torque, -self.effort_limit, self.effort_limit)

        limited = np.asarray(self.model.actuator_ctrllimited[self.actuator_ids_np], dtype=bool)
        if np.any(limited):
            ctrlrange = self.model.actuator_ctrlrange[self.actuator_ids_np]
            torque[limited] = np.clip(torque[limited], ctrlrange[limited, 0], ctrlrange[limited, 1])

        self.data.ctrl[self.actuator_ids_np] = torque
        self.last_torque = torque.copy()
        return self.last_torque

    def _apply_elastic_band(self) -> None:
        if not self.elastic_band or self.elastic_body_id is None:
            return
        self.data.xfrc_applied[self.elastic_body_id, :] = 0.0
        pos = self.data.xpos[self.elastic_body_id]
        lin_vel = self.data.cvel[self.elastic_body_id, 3:6]
        delta = self.elastic_point - pos
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8:
            return
        direction = delta / distance
        velocity_along_band = float(np.dot(lin_vel, direction))
        force = (
            self.elastic_stiffness * (distance - self.elastic_length)
            - self.elastic_damping * velocity_along_band
        ) * direction
        self.data.xfrc_applied[self.elastic_body_id, :3] = force

    def step(self, q_target: np.ndarray) -> MujocoSimStatus:
        for _ in range(self.sim_substeps):
            self._apply_elastic_band()
            self.apply_q_target(q_target)
            self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        return self.status()

    def step_damping(self) -> MujocoSimStatus:
        self.data.xfrc_applied[:] = 0.0
        for _ in range(self.sim_substeps):
            self.apply_damping()
            self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        return self.status()

    def status(self) -> MujocoSimStatus:
        return MujocoSimStatus(
            time=float(self.data.time),
            root_pos=self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3].copy(),
            root_quat_wxyz=self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7].copy(),
            dof_pos=self.data.qpos[self.qpos_adrs_np].copy(),
            dof_vel=self.data.qvel[self.qvel_adrs_np].copy(),
            torque=self.last_torque.copy(),
        )

    def is_running(self) -> bool:
        return self.viewer is None or bool(self.viewer.is_running())

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._floor_context is not None:
            self._floor_context.__exit__(None, None, None)
            self._floor_context = None
        self._prepared_context.__exit__(None, None, None)
