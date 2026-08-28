"""Run the GMR retargeting viewer in a separate process."""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any

import numpy as np


def _jsonable_human_motion(
    human_motion_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, tuple[list[float], list[float]]] | None:
    if human_motion_data is None:
        return None
    return {
        str(name): (
            [float(v) for v in np.asarray(pos, dtype=np.float64).reshape(3)],
            [float(v) for v in np.asarray(rot, dtype=np.float64).reshape(4)],
        )
        for name, (pos, rot) in human_motion_data.items()
    }


def _numpy_human_motion(
    payload: dict[str, tuple[list[float], list[float]]] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    if payload is None:
        return None
    return {
        str(name): (
            np.asarray(pos, dtype=np.float64).reshape(3),
            np.asarray(rot, dtype=np.float64).reshape(4),
        )
        for name, (pos, rot) in payload.items()
    }


def _worker(robot: str, hz: float, show_human: bool, data_queue: mp.Queue) -> None:
    from kitov_deploy.gmr_online import make_robot_motion_viewer

    viewer = make_robot_motion_viewer(robot, motion_fps=hz)
    try:
        while True:
            payload = data_queue.get()
            if payload is None:
                break
            qpos = np.asarray(payload["qpos"], dtype=np.float64).reshape(-1)
            human_motion = _numpy_human_motion(payload.get("human_motion")) if show_human else None
            viewer.step_qpos(qpos, human_motion_data=human_motion, rate_limit=False)
    finally:
        viewer.close()


class GMRViewerProcess:
    def __init__(
        self,
        robot: str,
        *,
        gmr_root: Any = None,
        hz: float = 50.0,
        show_human: bool = False,
    ) -> None:
        del gmr_root
        self.show_human = bool(show_human)
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=1)
        self._process = self._ctx.Process(
            target=_worker,
            args=(str(robot), float(hz), self.show_human, self._queue),
            daemon=True,
        )
        self._process.start()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def publish(
        self,
        qpos: np.ndarray,
        *,
        human_motion_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        if not self.is_alive():
            return
        payload: dict[str, Any] = {
            "qpos": [float(v) for v in np.asarray(qpos, dtype=np.float64).reshape(-1)],
        }
        if self.show_human:
            payload["human_motion"] = _jsonable_human_motion(human_motion_data)

        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        if self.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
