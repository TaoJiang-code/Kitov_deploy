"""Live XRobot/PICO body stream conversion for online retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


XR_BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]


_UNITY_TO_RHS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_UNITY_TO_RHS_QUAT = R.from_matrix(_UNITY_TO_RHS).as_quat(scalar_first=True)


@dataclass(frozen=True)
class XRobotBodyFrame:
    body: dict[str, tuple[np.ndarray, np.ndarray]]
    timestamp_ns: int
    raw_poses: dict[str, list[float]]


@dataclass(frozen=True)
class XRobotControllerState:
    primary_button: bool
    secondary_button: bool
    axis_click: bool
    trigger: float
    grip: float
    axis: tuple[float, float]


@dataclass(frozen=True)
class XRobotControllerFrame:
    controllers: dict[str, XRobotControllerState]
    timestamp_ns: int


def body_frame_from_raw_poses(
    raw_poses: dict[str, list[float]],
    *,
    timestamp_ns: int = 0,
) -> XRobotBodyFrame:
    """Convert raw Unity pose arrays into the body format consumed by GMR.

    The relay protocol intentionally carries the raw Unity coordinates. This
    function is shared by the local SDK reader and the remote UDP reader so
    the coordinate conversion happens exactly once on the GMR machine.
    """

    body: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    normalized_raw: dict[str, list[float]] = {}
    for joint_name in XR_BODY_JOINT_NAMES:
        pose = raw_poses.get(joint_name)
        if not isinstance(pose, (list, tuple)) or len(pose) < 7:
            continue
        x, y, z, qx, qy, qz, qw = [float(value) for value in pose[:7]]
        normalized_raw[joint_name] = [x, y, z, qx, qy, qz, qw]
        pos = np.array([x, y, z], dtype=np.float64) @ _UNITY_TO_RHS.T
        quat_wxyz = quat_mul_wxyz(
            _UNITY_TO_RHS_QUAT,
            np.array([qw, qx, qy, qz], dtype=np.float64),
        )
        body[joint_name] = (pos, quat_wxyz)
    return XRobotBodyFrame(
        body=body,
        timestamp_ns=int(timestamp_ns),
        raw_poses=normalized_raw,
    )


def _load_xrobot_sdk() -> Any:
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import xrobotoolkit_sdk. Activate the Kitov_deploy uv "
            "environment and install the XRoboToolkit Python binding first."
        ) from exc
    if not hasattr(xrt, "init"):
        raise RuntimeError("Installed xrobotoolkit_sdk does not expose init().")
    return xrt


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication with scalar-first [w, x, y, z] layout."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _safe_xrt_call(xrt: Any, name: str, default: Any = None) -> Any:
    fn = getattr(xrt, name, None)
    if fn is None:
        return default
    try:
        return fn()
    except Exception:
        return default


def _axis(values: Any) -> tuple[float, float]:
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        try:
            return (float(values[0]), float(values[1]))
        except Exception:
            return (0.0, 0.0)
    return (0.0, 0.0)


class XRobotBodyStreamer:
    """Polling reader that emits GMR-compatible body dictionaries.

    Raw XRobot body poses are `[x, y, z, qx, qy, qz, qw]` in Unity coordinates.
    GMR expects right-handed coordinates and scalar-first quaternions
    `[qw, qx, qy, qz]`.
    """

    def __init__(self) -> None:
        self._xrt = _load_xrobot_sdk()

    def start(self) -> None:
        self._xrt.init()

    def close(self) -> None:
        close = getattr(self._xrt, "close", None)
        if close is not None:
            close()

    def read_body_frame(self) -> XRobotBodyFrame | None:
        raw_frame = self.read_raw_body_frame()
        if raw_frame is None:
            return None
        return body_frame_from_raw_poses(raw_frame.raw_poses, timestamp_ns=raw_frame.timestamp_ns)

    def read_raw_body_frame(self) -> XRobotBodyFrame | None:
        """Read a frame without applying the Unity-to-GMR coordinate transform."""

        if not self._xrt.is_body_data_available():
            return None

        raw_poses = self._xrt.get_body_joints_pose()
        timestamp_ns = int(getattr(self._xrt, "get_body_timestamp_ns")() or 0)
        if not isinstance(raw_poses, (list, tuple)) or len(raw_poses) < len(XR_BODY_JOINT_NAMES):
            return None

        raw_body: dict[str, list[float]] = {}
        for index, joint_name in enumerate(XR_BODY_JOINT_NAMES):
            pose = raw_poses[index]
            if not isinstance(pose, (list, tuple)) or len(pose) < 7:
                continue
            x, y, z, qx, qy, qz, qw = [float(v) for v in pose[:7]]
            raw_body[joint_name] = [x, y, z, qx, qy, qz, qw]
        return XRobotBodyFrame(body={}, timestamp_ns=timestamp_ns, raw_poses=raw_body)

    def read_controller_frame(self) -> XRobotControllerFrame:
        timestamp_ns = int(_safe_xrt_call(self._xrt, "get_time_stamp_ns", 0) or 0)
        return XRobotControllerFrame(
            controllers={
                "left": XRobotControllerState(
                    primary_button=bool(_safe_xrt_call(self._xrt, "get_X_button", False)),
                    secondary_button=bool(_safe_xrt_call(self._xrt, "get_Y_button", False)),
                    axis_click=bool(_safe_xrt_call(self._xrt, "get_left_axis_click", False)),
                    trigger=float(_safe_xrt_call(self._xrt, "get_left_trigger", 0.0) or 0.0),
                    grip=float(_safe_xrt_call(self._xrt, "get_left_grip", 0.0) or 0.0),
                    axis=_axis(_safe_xrt_call(self._xrt, "get_left_axis", [0.0, 0.0])),
                ),
                "right": XRobotControllerState(
                    primary_button=bool(_safe_xrt_call(self._xrt, "get_A_button", False)),
                    secondary_button=bool(_safe_xrt_call(self._xrt, "get_B_button", False)),
                    axis_click=bool(_safe_xrt_call(self._xrt, "get_right_axis_click", False)),
                    trigger=float(_safe_xrt_call(self._xrt, "get_right_trigger", 0.0) or 0.0),
                    grip=float(_safe_xrt_call(self._xrt, "get_right_grip", 0.0) or 0.0),
                    axis=_axis(_safe_xrt_call(self._xrt, "get_right_axis", [0.0, 0.0])),
                ),
            },
            timestamp_ns=timestamp_ns,
        )
