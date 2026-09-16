"""Local XRobot body retargeting and visualization for Kitov deploy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kitov_deploy.mjcf_utils import prepared_mjcf_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMR_ROOT: Path | None = None


@dataclass(frozen=True)
class RobotRetargetConfig:
    name: str
    xml_path: Path
    ik_config_path: Path
    viewer_base_body: str
    viewer_camera_distance: float = 2.0
    human_joint_aliases: dict[str, str] | None = None

    @property
    def gmr_robot(self) -> str:
        return self.name


ROBOT_CONFIGS = {
    "g1": RobotRetargetConfig(
        name="g1",
        xml_path=REPO_ROOT / "Glush_Zoo" / "g1_description" / "mjcf" / "scene_29dof.xml",
        ik_config_path=REPO_ROOT / "configs" / "gmr" / "xrobot_to_g1.json",
        viewer_base_body="pelvis",
        viewer_camera_distance=2.0,
    ),
    "unitree_g1": RobotRetargetConfig(
        name="g1",
        xml_path=REPO_ROOT / "Glush_Zoo" / "g1_description" / "mjcf" / "scene_29dof.xml",
        ik_config_path=REPO_ROOT / "configs" / "gmr" / "xrobot_to_g1.json",
        viewer_base_body="pelvis",
        viewer_camera_distance=2.0,
    ),
    "bumi": RobotRetargetConfig(
        name="bumi",
        xml_path=REPO_ROOT / "Glush_Zoo" / "bumi" / "mjcf" / "scene_21dof.xml",
        ik_config_path=REPO_ROOT / "configs" / "gmr" / "xrobot_to_bumi.json",
        viewer_base_body="base_link",
        viewer_camera_distance=2.0,
    ),
    "openarm": RobotRetargetConfig(
        name="openarm_v1",
        xml_path=REPO_ROOT / "Glush_Zoo" / "openarm_v1" / "mjcf" / "scene.xml",
        ik_config_path=REPO_ROOT / "configs" / "gmr" / "xrobot_to_openarm_v1.json",
        viewer_base_body="openarm_body_link0",
        viewer_camera_distance=3.0,
    ),
    "openarm_v1": RobotRetargetConfig(
        name="openarm_v1",
        xml_path=REPO_ROOT / "Glush_Zoo" / "openarm_v1" / "mjcf" / "scene.xml",
        ik_config_path=REPO_ROOT / "configs" / "gmr" / "xrobot_to_openarm_v1.json",
        viewer_base_body="openarm_body_link0",
        viewer_camera_distance=3.0,
    ),
}


def resolve_robot_config(robot: str) -> RobotRetargetConfig:
    key = robot.lower()
    if key not in ROBOT_CONFIGS:
        supported = ", ".join(sorted(ROBOT_CONFIGS))
        raise ValueError(f"Unsupported robot '{robot}'. Supported robots: {supported}")
    return ROBOT_CONFIGS[key]


def _load_retarget_dependencies() -> tuple[Any, Any, Any]:
    try:
        import mink
        import mujoco
        from scipy.spatial.transform import Rotation as Rotation
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import local GMR retargeting dependencies. Install them in "
            "the Kitov_deploy conda environment first:\n"
            "python -m pip install mujoco mink daqp 'qpsolvers[proxqp]' scipy"
        ) from exc
    return mink, mujoco, Rotation


def _load_viewer_dependencies() -> tuple[Any, Any]:
    try:
        import mujoco
        import mujoco.viewer
        from loop_rate_limiters import RateLimiter
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import local GMR viewer dependencies. Install them in the "
            "Kitov_deploy conda environment first:\n"
            "python -m pip install mujoco loop-rate-limiters"
        ) from exc
    return mujoco, RateLimiter


def _as_pose_dict(
    human_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (pos, quat) in human_data.items():
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(quat_arr)
        if norm > 1e-8:
            quat_arr = quat_arr / norm
        converted[str(name)] = (pos_arr, quat_arr)
    return converted


class LocalMotionRetargeting:
    """Mink-based retargeter kept inside Kitov_deploy.

    Quaternions use MuJoCo/GMR scalar-first order: w, x, y, z.
    """

    def __init__(
        self,
        *,
        xml_path: Path,
        ik_config_path: Path,
        actual_human_height: float | None = None,
        human_joint_aliases: dict[str, str] | None = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = True,
        use_velocity_limit: bool = False,
    ) -> None:
        self.mink, self.mujoco, self.Rotation = _load_retarget_dependencies()
        self.xml_path = Path(xml_path).expanduser()
        self.ik_config_path = Path(ik_config_path).expanduser()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {self.xml_path}")
        if not self.ik_config_path.exists():
            raise FileNotFoundError(f"IK config not found: {self.ik_config_path}")

        with prepared_mjcf_path(self.xml_path) as xml_path:
            self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self.robot_dof_names = {
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, i): i
            for i in range(self.model.nv)
        }
        self.robot_body_names = {
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, i): i
            for i in range(self.model.nbody)
        }
        self.robot_motor_names = {
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }
        self.has_floating_base = bool(np.any(self.model.jnt_type == self.mujoco.mjtJoint.mjJNT_FREE))

        ik_config = json.loads(self.ik_config_path.read_text(encoding="utf-8"))
        self.ik_match_table1 = dict(ik_config["ik_match_table1"])
        self.ik_match_table2 = dict(ik_config["ik_match_table2"])
        self.target_origin = dict(ik_config.get("target_origin", {}))
        self.human_joint_aliases = dict(human_joint_aliases or {})
        self.human_root_name = str(ik_config["human_root_name"])
        self.robot_root_name = str(ik_config["robot_root_name"])
        self.use_ik_match_table1 = bool(ik_config["use_ik_match_table1"])
        self.use_ik_match_table2 = bool(ik_config["use_ik_match_table2"])
        self.ground = float(ik_config["ground_height"]) * np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.max_iter = int(ik_config.get("max_iter", 10))
        self.solver = solver
        self.damping = float(damping)
        self.ground_offset = 0.0
        self.scaled_human_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        human_scale_table = dict(ik_config["human_scale_table"])
        if actual_human_height is not None:
            assumed = float(ik_config["human_height_assumption"])
            human_scale_table = {
                key: float(value) * float(actual_human_height) / assumed
                for key, value in human_scale_table.items()
            }
        self.human_scale_table = human_scale_table

        self._validate_ik_config()
        self._setup_retarget_configuration()

        self.ik_limits = [self.mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            velocity_limits = {name: 3.14 for name in self.robot_motor_names if name}
            self.ik_limits.append(self.mink.VelocityLimit(self.model, velocity_limits))

        if verbose:
            self._print_model_summary()

    def _validate_ik_config(self) -> None:
        missing_bodies = sorted(
            {
                robot_body
                for table in (self.ik_match_table1, self.ik_match_table2)
                for robot_body in table
                if robot_body not in self.robot_body_names
            }
        )
        if missing_bodies:
            raise KeyError(
                f"IK config {self.ik_config_path} references missing robot bodies "
                f"in {self.xml_path}: {missing_bodies}"
            )

        human_names = {
            spec[0]
            for table in (self.ik_match_table1, self.ik_match_table2)
            for spec in table.values()
        }
        missing_scales = sorted(name for name in human_names if name not in self.human_scale_table)
        if missing_scales:
            raise KeyError(f"IK config {self.ik_config_path} has no human_scale_table entries for: {missing_scales}")

    def _print_model_summary(self) -> None:
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        for name, idx in self.robot_dof_names.items():
            print(f"[GMR]   {idx}: {name}")
        print("[GMR] Robot Body names and their IDs:")
        for name, idx in self.robot_body_names.items():
            print(f"[GMR]   {idx}: {name}")
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        for name, idx in self.robot_motor_names.items():
            print(f"[GMR]   {idx}: {name}")

    def _setup_retarget_configuration(self) -> None:
        self.configuration = self.mink.Configuration(self.model)
        self.tasks1: list[Any] = []
        self.tasks2: list[Any] = []
        self.task_errors1: list[np.ndarray] = []
        self.task_errors2: list[np.ndarray] = []
        self.ik_match_table1_task_map: dict[str, Any] = {}
        self.ik_match_table2_task_map: dict[str, Any] = {}
        self.ik_match_table1_human_body_to_task: dict[str, str] = {}
        self.ik_match_table2_human_body_to_task: dict[str, str] = {}
        self.ik_match_table1_pos_offsets: dict[str, np.ndarray] = {}
        self.ik_match_table2_pos_offsets: dict[str, np.ndarray] = {}
        self.ik_match_table1_rot_offsets: dict[str, Any] = {}
        self.ik_match_table2_rot_offsets: dict[str, Any] = {}

        if self.use_ik_match_table1:
            self._setup_tasks(
                self.ik_match_table1,
                self.tasks1,
                self.ik_match_table1_task_map,
                self.ik_match_table1_human_body_to_task,
                self.ik_match_table1_pos_offsets,
                self.ik_match_table1_rot_offsets,
            )
        if self.use_ik_match_table2:
            self._setup_tasks(
                self.ik_match_table2,
                self.tasks2,
                self.ik_match_table2_task_map,
                self.ik_match_table2_human_body_to_task,
                self.ik_match_table2_pos_offsets,
                self.ik_match_table2_rot_offsets,
            )

    def _setup_tasks(
        self,
        table: dict[str, list[Any]],
        tasks: list[Any],
        task_map: dict[str, Any],
        human_body_to_task: dict[str, str],
        pos_offsets: dict[str, np.ndarray],
        rot_offsets: dict[str, Any],
    ) -> None:
        for robot_body, (human_body, pos_weight, rot_weight, pos_offset, rot_offset) in table.items():
            task = self.mink.FrameTask(
                frame_name=robot_body,
                frame_type="body",
                position_cost=float(pos_weight),
                orientation_cost=float(rot_weight),
                lm_damping=1.0,
            )
            tasks.append(task)
            task_map[robot_body] = task
            human_body_to_task[str(human_body)] = robot_body
            pos_offsets[str(human_body)] = np.asarray(pos_offset, dtype=np.float64).reshape(3)
            rot_offsets[str(human_body)] = self.Rotation.from_quat(rot_offset, scalar_first=True)

    def _scale_human_data(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        include_unscaled: bool = False,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.human_root_name not in human_data:
            raise KeyError(f"Human data missing root joint {self.human_root_name!r}")
        root_pos = human_data[self.human_root_name][0]
        scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        names = human_data.keys() if include_unscaled else self.human_scale_table.keys()
        for name in names:
            if name not in human_data:
                continue
            scale = self.human_scale_table.get(name, 1.0)
            pos, quat = human_data[name]
            if name == self.human_root_name:
                scaled_pos = pos if self.has_floating_base else pos * float(scale)
            else:
                scaled_pos = (pos - root_pos) * float(scale) + root_pos
            scaled[name] = (scaled_pos, quat.copy())
        return scaled

    def _apply_human_joint_aliases(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not self.human_joint_aliases:
            return human_data
        aliased = dict(human_data)
        for target_name, source_name in self.human_joint_aliases.items():
            if target_name in aliased or source_name not in human_data:
                continue
            pos, quat = human_data[source_name]
            aliased[target_name] = (pos.copy(), quat.copy())
        return aliased

    def _extract_yaw_rotation(self, quat: np.ndarray) -> Any:
        root_rot = self.Rotation.from_quat(quat, scalar_first=True)
        forward = root_rot.apply(np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(forward[:2]) < 1e-8:
            return self.Rotation.identity()
        yaw = np.arctan2(forward[1], forward[0])
        return self.Rotation.from_euler("z", yaw)

    def _align_fixed_base_arm_targets(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not self.target_origin:
            return human_data

        human_origin_name = str(self.target_origin.get("human_origin", "Spine2"))
        root_rotation_name = str(self.target_origin.get("root_rotation_body", human_origin_name))
        robot_origin_name = str(self.target_origin.get("robot_origin", self.robot_root_name))
        lock_root_yaw = bool(self.target_origin.get("lock_root_yaw", False))
        heading_yaw_offset_deg = float(self.target_origin.get("heading_yaw_offset_deg", 0.0))
        if human_origin_name not in human_data:
            raise KeyError(f"OpenArm human_origin {human_origin_name!r} is missing from human data")
        if lock_root_yaw and root_rotation_name not in human_data:
            raise KeyError(f"OpenArm root_rotation_body {root_rotation_name!r} is missing from human data")

        robot_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, robot_origin_name)
        if robot_body_id == -1:
            raise ValueError(f"OpenArm robot_origin {robot_origin_name!r} is not a valid robot body")

        self.mujoco.mj_forward(self.model, self.configuration.data)
        human_origin_pos = human_data[human_origin_name][0]
        human_origin_rot = self.Rotation.from_quat(human_data[human_origin_name][1], scalar_first=True)
        robot_origin_pos = self.configuration.data.xpos[robot_body_id].copy()
        robot_origin_rot = self.Rotation.from_quat(self.configuration.data.xquat[robot_body_id].copy(), scalar_first=True)
        local_offset = np.asarray(self.target_origin.get("position_offset", [0.0, 0.0, 0.698]), dtype=np.float64)
        rotation_offset = self.Rotation.from_quat(
            self.target_origin.get("rotation_offset", [1.0, 0.0, 0.0, 0.0]),
            scalar_first=True,
        )
        heading_yaw_offset = self.Rotation.from_euler("z", np.deg2rad(heading_yaw_offset_deg))
        if lock_root_yaw:
            inv_root_yaw = self._extract_yaw_rotation(human_data[root_rotation_name][1]).inv()
            target_origin_rot = robot_origin_rot * heading_yaw_offset * rotation_offset
            target_origin_pos = robot_origin_pos + target_origin_rot.apply(local_offset)
        else:
            inv_root_yaw = self.Rotation.identity()
            target_origin_rot = human_origin_rot * heading_yaw_offset * rotation_offset * human_origin_rot.inv()
            target_origin_pos = robot_origin_pos + local_offset

        aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (pos, quat) in human_data.items():
            local_pos = inv_root_yaw.apply(pos - human_origin_pos)
            local_quat = inv_root_yaw * self.Rotation.from_quat(quat, scalar_first=True)
            aligned[name] = (
                target_origin_pos + target_origin_rot.apply(local_pos),
                (target_origin_rot * local_quat).as_quat(scalar_first=True),
            )
        return aligned

    def _offset_human_data(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        pos_offsets: dict[str, np.ndarray],
        rot_offsets: dict[str, Any],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        offset_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (pos, quat) in human_data.items():
            rot = self.Rotation.from_quat(quat, scalar_first=True)
            if name in rot_offsets:
                rot = rot * rot_offsets[name]
            updated_quat = rot.as_quat(scalar_first=True)
            updated_pos = pos.copy()
            if name in pos_offsets:
                updated_pos = pos + rot.apply(pos_offsets[name])
            offset_data[name] = (updated_pos, updated_quat)
        return offset_data

    def _apply_ground_offset(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.ground_offset == 0.0:
            return human_data
        offset = np.array([0.0, 0.0, self.ground_offset], dtype=np.float64)
        return {name: (pos - offset, quat) for name, (pos, quat) in human_data.items()}

    def _offset_human_data_to_ground(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        ground_clearance: float = 0.1,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        lowest = None
        for name, (pos, _quat) in human_data.items():
            lower = name.lower()
            if "foot" not in lower and "toe" not in lower:
                continue
            lowest = float(pos[2]) if lowest is None else min(lowest, float(pos[2]))
        if lowest is None:
            return human_data
        offset = np.array([0.0, 0.0, lowest - ground_clearance], dtype=np.float64)
        return {name: (pos - offset, quat) for name, (pos, quat) in human_data.items()}

    def set_ground_offset(self, value: float) -> None:
        self.ground_offset = float(value)

    def update_targets(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        offset_to_ground: bool = False,
    ) -> None:
        human_data = _as_pose_dict(human_data)
        human_data = self._apply_human_joint_aliases(human_data)
        scaled_data = self._scale_human_data(human_data)
        scaled_data = self._align_fixed_base_arm_targets(scaled_data)
        scaled_data = self._apply_ground_offset(scaled_data)
        if offset_to_ground:
            scaled_data = self._offset_human_data_to_ground(scaled_data)
        self.scaled_human_data = scaled_data

        if self.use_ik_match_table1:
            self._update_task_targets(
                scaled_data,
                self.ik_match_table1_human_body_to_task,
                self.ik_match_table1_pos_offsets,
                self.ik_match_table1_rot_offsets,
                self.ik_match_table1_task_map,
            )
        if self.use_ik_match_table2:
            self._update_task_targets(
                scaled_data,
                self.ik_match_table2_human_body_to_task,
                self.ik_match_table2_pos_offsets,
                self.ik_match_table2_rot_offsets,
                self.ik_match_table2_task_map,
            )

    def prepare_debug_human_data(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        offset_to_ground: bool = False,
        include_unscaled: bool = False,
        task_table: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        human_data = _as_pose_dict(human_data)
        human_data = self._apply_human_joint_aliases(human_data)
        scaled_data = self._scale_human_data(human_data, include_unscaled=include_unscaled)
        scaled_data = self._align_fixed_base_arm_targets(scaled_data)
        if task_table is not None:
            if task_table == "ik_match_table1":
                scaled_data = self._offset_human_data(
                    scaled_data,
                    self.ik_match_table1_pos_offsets,
                    self.ik_match_table1_rot_offsets,
                )
            elif task_table == "ik_match_table2":
                scaled_data = self._offset_human_data(
                    scaled_data,
                    self.ik_match_table2_pos_offsets,
                    self.ik_match_table2_rot_offsets,
                )
            else:
                raise ValueError(f"Unknown IK task table: {task_table}")
        scaled_data = self._apply_ground_offset(scaled_data)
        if offset_to_ground:
            scaled_data = self._offset_human_data_to_ground(scaled_data)
        return scaled_data

    def _update_task_targets(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        human_body_to_task: dict[str, str],
        pos_offsets: dict[str, np.ndarray],
        rot_offsets: dict[str, Any],
        task_map: dict[str, Any],
    ) -> None:
        offset_data = self._offset_human_data(human_data, pos_offsets, rot_offsets)
        for human_body, robot_body in human_body_to_task.items():
            if human_body not in offset_data:
                continue
            pos, quat = offset_data[human_body]
            task_map[robot_body].set_target(
                self.mink.SE3.from_rotation_and_translation(self.mink.SO3(quat), pos - self.ground)
            )

    def _task_error_norm(self, tasks: list[Any]) -> float:
        errors = [task.compute_error(self.configuration) for task in tasks]
        if not errors:
            return 0.0
        return float(np.linalg.norm(np.concatenate(errors)))

    def retarget(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        offset_to_ground: bool = False,
    ) -> np.ndarray:
        self.update_targets(human_data, offset_to_ground=offset_to_ground)

        if self.use_ik_match_table1:
            curr_error = self._task_error_norm(self.tasks1)
            dt = self.configuration.model.opt.timestep
            vel1 = self.mink.solve_ik(
                self.configuration,
                self.tasks1,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self._task_error_norm(self.tasks1)
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                vel1 = self.mink.solve_ik(
                    self.configuration,
                    self.tasks1,
                    dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self._task_error_norm(self.tasks1)
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self._task_error_norm(self.tasks2)
            dt = self.configuration.model.opt.timestep
            vel2 = self.mink.solve_ik(
                self.configuration,
                self.tasks2,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self._task_error_norm(self.tasks2)
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                vel2 = self.mink.solve_ik(
                    self.configuration,
                    self.tasks2,
                    dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self._task_error_norm(self.tasks2)
                num_iter += 1

        return self.configuration.data.qpos.copy()


def _draw_frame(viewer: Any, position: np.ndarray, rotation_matrix: np.ndarray, *, alpha: float = 1.0) -> None:
    import mujoco as mj

    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]
    scale = 0.1
    arrow_size = 0.01

    def _arrow(axis: np.ndarray, rgba: np.ndarray) -> None:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_ARROW,
            np.array([arrow_size, arrow_size, scale], dtype=np.float64),
            position,
            np.eye(3).flatten(),
            rgba,
        )
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_ARROW, arrow_size, position, position + scale * axis)
        viewer.user_scn.ngeom += 1

    _arrow(x_axis, np.array([1.0, 0.0, 0.0, alpha]))
    _arrow(y_axis, np.array([0.0, 1.0, 0.0, alpha]))
    _arrow(z_axis, np.array([0.0, 0.0, 1.0, alpha]))


class LocalRobotMotionViewer:
    def __init__(
        self,
        *,
        xml_path: Path,
        robot_base: str,
        viewer_cam_distance: float,
        motion_fps: float = 30.0,
        transparent_robot: float = 0.0,
        keyboard_callback: Any | None = None,
    ) -> None:
        self.mujoco, RateLimiter = _load_viewer_dependencies()
        self.xml_path = Path(xml_path).expanduser()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {self.xml_path}")
        with prepared_mjcf_path(self.xml_path) as xml_path:
            self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self.mujoco.MjData(self.model)
        self.robot_base = robot_base
        self.rate_limiter = RateLimiter(frequency=motion_fps, warn=False)

        if transparent_robot:
            self.model.geom_rgba[:, 3] = float(transparent_robot)

        self.viewer = self.mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=True,
            show_right_ui=True,
            key_callback=keyboard_callback,
        )
        self.viewer.cam.distance = float(viewer_cam_distance)
        self.viewer.cam.elevation = -20

    def step(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        dof_pos: np.ndarray,
        *,
        human_motion_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        show_human_body_name: bool = False,
        show_human_points: bool = True,
        human_point_scale: float = 0.1,
        human_pos_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
        rate_limit: bool = True,
        follow_camera: bool = True,
    ) -> None:
        qpos = np.concatenate(
            [
                np.asarray(root_pos, dtype=np.float64).reshape(3),
                np.asarray(root_rot, dtype=np.float64).reshape(4),
                np.asarray(dof_pos, dtype=np.float64).reshape(-1),
            ]
        )
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos size mismatch for {self.xml_path}: got {qpos.shape[0]}, expected {self.model.nq}")
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)

        if follow_camera:
            try:
                body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.robot_base)
                self.viewer.cam.lookat = self.data.xpos[body_id]
            except Exception:
                pass

        if human_motion_data is not None:
            self.viewer.user_scn.ngeom = 0
            for name, (pos, quat) in human_motion_data.items():
                pos = np.asarray(pos, dtype=np.float64).reshape(3) + human_pos_offset
                quat = np.asarray(quat, dtype=np.float64).reshape(4)
                mat = np.zeros(9, dtype=np.float64)
                self.mujoco.mju_quat2Mat(mat, quat)
                _draw_frame(self.viewer, pos, mat.reshape(3, 3), alpha=1.0)

                if show_human_points:
                    geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                    self.mujoco.mjv_initGeom(
                        geom,
                        self.mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([human_point_scale, 0.0, 0.0], dtype=np.float64),
                        pos,
                        np.eye(3).flatten(),
                        np.array([0.2, 0.2, 1.0, 0.5]),
                    )
                    self.viewer.user_scn.ngeom += 1

                if show_human_body_name:
                    label = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                    self.mujoco.mjv_initGeom(
                        label,
                        self.mujoco.mjtGeom.mjGEOM_LABEL,
                        np.array([0.0, 0.0, 0.0], dtype=np.float64),
                        pos + np.array([0.0, 0.0, 0.05]),
                        np.eye(3).flatten(),
                        np.array([1.0, 1.0, 1.0, 1.0]),
                    )
                    label.label = str(name)
                    self.viewer.user_scn.ngeom += 1

        self.viewer.sync()
        if rate_limit:
            self.rate_limiter.sleep()

    def step_qpos(
        self,
        qpos: np.ndarray,
        *,
        human_motion_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        show_human_body_name: bool = False,
        show_human_points: bool = True,
        human_point_scale: float = 0.1,
        human_pos_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
        rate_limit: bool = True,
        follow_camera: bool = True,
    ) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos size mismatch for {self.xml_path}: got {qpos.shape[0]}, expected {self.model.nq}")
        self.data.qpos[:] = qpos
        self.mujoco.mj_forward(self.model, self.data)

        if follow_camera:
            try:
                body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.robot_base)
                self.viewer.cam.lookat = self.data.xpos[body_id]
            except Exception:
                pass

        if human_motion_data is not None:
            self._draw_human_motion(
                human_motion_data,
                show_human_body_name=show_human_body_name,
                show_human_points=show_human_points,
                human_point_scale=human_point_scale,
                human_pos_offset=human_pos_offset,
            )

        self.viewer.sync()
        if rate_limit:
            self.rate_limiter.sleep()

    def _draw_human_motion(
        self,
        human_motion_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        show_human_body_name: bool = False,
        show_human_points: bool = True,
        human_point_scale: float = 0.1,
        human_pos_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> None:
        self.viewer.user_scn.ngeom = 0
        for name, (pos, quat) in human_motion_data.items():
            pos = np.asarray(pos, dtype=np.float64).reshape(3) + human_pos_offset
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            mat = np.zeros(9, dtype=np.float64)
            self.mujoco.mju_quat2Mat(mat, quat)
            _draw_frame(self.viewer, pos, mat.reshape(3, 3), alpha=1.0)

            if show_human_points:
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                self.mujoco.mjv_initGeom(
                    geom,
                    self.mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([human_point_scale, 0.0, 0.0], dtype=np.float64),
                    pos,
                    np.eye(3).flatten(),
                    np.array([0.2, 0.2, 1.0, 0.5]),
                )
                self.viewer.user_scn.ngeom += 1

            if show_human_body_name:
                label = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                self.mujoco.mjv_initGeom(
                    label,
                    self.mujoco.mjtGeom.mjGEOM_LABEL,
                    np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    pos + np.array([0.0, 0.0, 0.05]),
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 1.0, 1.0]),
                )
                label.label = str(name)
                self.viewer.user_scn.ngeom += 1

    def close(self) -> None:
        self.viewer.close()


def make_robot_motion_viewer(
    robot: str,
    *,
    motion_fps: float = 50.0,
    transparent_robot: float = 0.0,
    keyboard_callback: Any | None = None,
) -> LocalRobotMotionViewer:
    config = resolve_robot_config(robot)
    return LocalRobotMotionViewer(
        xml_path=config.xml_path,
        robot_base=config.viewer_base_body,
        viewer_cam_distance=config.viewer_camera_distance,
        motion_fps=motion_fps,
        transparent_robot=transparent_robot,
        keyboard_callback=keyboard_callback,
    )


class OnlineGMRRetargeter:
    def __init__(
        self,
        robot: str,
        *,
        gmr_root: Path | None = DEFAULT_GMR_ROOT,
        actual_human_height: float | None = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        use_velocity_limit: bool = False,
        ik_config_path: Path | None = None,
        verbose: bool = True,
    ) -> None:
        del gmr_root
        self.config = resolve_robot_config(robot)
        resolved_ik_config_path = self.config.ik_config_path if ik_config_path is None else Path(ik_config_path)
        self._retargeter = LocalMotionRetargeting(
            xml_path=self.config.xml_path,
            ik_config_path=resolved_ik_config_path,
            actual_human_height=actual_human_height,
            human_joint_aliases=self.config.human_joint_aliases,
            solver=solver,
            damping=damping,
            verbose=verbose,
            use_velocity_limit=use_velocity_limit,
        )

    @property
    def robot_dof_names(self) -> dict[str, int]:
        return self._retargeter.robot_dof_names

    @property
    def robot_body_names(self) -> dict[str, int]:
        return self._retargeter.robot_body_names

    @property
    def scaled_human_data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._retargeter.scaled_human_data

    @property
    def robot_qpos_size(self) -> int:
        return int(self._retargeter.model.nq)

    @property
    def robot_dof_size(self) -> int:
        return int(self._retargeter.model.nv)

    @property
    def has_floating_base(self) -> bool:
        return bool(self._retargeter.has_floating_base)

    def set_ground_offset(self, value: float) -> None:
        self._retargeter.set_ground_offset(value)

    def retarget(self, human_body: dict[str, tuple[np.ndarray, np.ndarray]], *, offset_to_ground: bool = False) -> np.ndarray:
        return self._retargeter.retarget(human_body, offset_to_ground=offset_to_ground)

    def prepare_debug_human_data(
        self,
        human_body: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        offset_to_ground: bool = False,
        include_unscaled: bool = False,
        task_table: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._retargeter.prepare_debug_human_data(
            human_body,
            offset_to_ground=offset_to_ground,
            include_unscaled=include_unscaled,
            task_table=task_table,
        )

    def make_viewer(
        self,
        *,
        motion_fps: float = 50.0,
        transparent_robot: float = 0.0,
        keyboard_callback: Any | None = None,
    ) -> LocalRobotMotionViewer:
        return make_robot_motion_viewer(
            self.config.name,
            motion_fps=motion_fps,
            transparent_robot=transparent_robot,
            keyboard_callback=keyboard_callback,
        )
