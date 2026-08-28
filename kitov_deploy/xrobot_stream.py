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


@dataclass(frozen=True)
class XRobotBodyFrame:
    body: dict[str, tuple[np.ndarray, np.ndarray]]
    timestamp_ns: int
    raw_poses: dict[str, list[float]]


def _load_xrobot_sdk() -> Any:
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import xrobotoolkit_sdk. Activate the Kitov_deploy conda "
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


class XRobotBodyStreamer:
    """Polling reader that emits GMR-compatible body dictionaries.

    Raw XRobot body poses are `[x, y, z, qx, qy, qz, qw]` in Unity coordinates.
    GMR expects right-handed coordinates and scalar-first quaternions
    `[qw, qx, qy, qz]`.
    """

    def __init__(self) -> None:
        self._xrt = _load_xrobot_sdk()
        self._unity_to_rhs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._unity_to_rhs_quat = R.from_matrix(self._unity_to_rhs).as_quat(scalar_first=True)

    def start(self) -> None:
        self._xrt.init()

    def close(self) -> None:
        close = getattr(self._xrt, "close", None)
        if close is not None:
            close()

    def read_body_frame(self) -> XRobotBodyFrame | None:
        if not self._xrt.is_body_data_available():
            return None

        raw_poses = self._xrt.get_body_joints_pose()
        timestamp_ns = int(getattr(self._xrt, "get_body_timestamp_ns")() or 0)
        if not isinstance(raw_poses, (list, tuple)) or len(raw_poses) < len(XR_BODY_JOINT_NAMES):
            return None

        body: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw_body: dict[str, list[float]] = {}
        for index, joint_name in enumerate(XR_BODY_JOINT_NAMES):
            pose = raw_poses[index]
            if not isinstance(pose, (list, tuple)) or len(pose) < 7:
                continue
            x, y, z, qx, qy, qz, qw = [float(v) for v in pose[:7]]
            raw_body[joint_name] = [x, y, z, qx, qy, qz, qw]
            pos = np.array([x, y, z], dtype=np.float64) @ self._unity_to_rhs.T
            quat_wxyz = quat_mul_wxyz(
                self._unity_to_rhs_quat,
                np.array([qw, qx, qy, qz], dtype=np.float64),
            )
            body[joint_name] = (pos, quat_wxyz)

        return XRobotBodyFrame(body=body, timestamp_ns=timestamp_ns, raw_poses=raw_body)
