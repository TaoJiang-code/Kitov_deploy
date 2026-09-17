"""UDP transport for forwarding raw XRobot/PICO frames to another machine."""

from __future__ import annotations

import json
import socket
import uuid
from dataclasses import asdict
from typing import Any

from .xrobot_stream import (
    XRobotBodyFrame,
    XRobotControllerFrame,
    XRobotControllerState,
    body_frame_from_raw_poses,
)


PROTOCOL_NAME = "kitov.xrobot.body.v1"


def encode_body_packet(
    body_frame: XRobotBodyFrame,
    controller_frame: XRobotControllerFrame | None = None,
    *,
    sequence: int,
    stream_id: str,
) -> bytes:
    """Encode one self-contained body frame as a small JSON UDP datagram."""

    controllers: dict[str, dict[str, Any]] = {}
    if controller_frame is not None:
        controllers = {
            name: asdict(state) for name, state in controller_frame.controllers.items()
        }
    packet = {
        "protocol": PROTOCOL_NAME,
        "stream_id": str(stream_id),
        "sequence": int(sequence),
        "timestamp_ns": int(body_frame.timestamp_ns),
        "controllers_timestamp_ns": int(controller_frame.timestamp_ns) if controller_frame else 0,
        "raw_poses": body_frame.raw_poses,
        "controllers": controllers,
    }
    return json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _controller_state(payload: Any) -> XRobotControllerState:
    if not isinstance(payload, dict):
        payload = {}
    axis = payload.get("axis", [0.0, 0.0])
    if not isinstance(axis, (list, tuple)) or len(axis) < 2:
        axis = [0.0, 0.0]
    return XRobotControllerState(
        primary_button=bool(payload.get("primary_button", False)),
        secondary_button=bool(payload.get("secondary_button", False)),
        axis_click=bool(payload.get("axis_click", False)),
        trigger=float(payload.get("trigger", 0.0)),
        grip=float(payload.get("grip", 0.0)),
        axis=(float(axis[0]), float(axis[1])),
    )


def decode_body_packet(
    payload: bytes,
) -> tuple[XRobotBodyFrame, XRobotControllerFrame, str, int]:
    """Decode and validate a relay packet.

    Returns ``body_frame, controller_frame, stream_id, sequence``. The caller
    is responsible for rejecting stale sequence numbers.
    """

    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict) or message.get("protocol") != PROTOCOL_NAME:
        raise ValueError("unsupported XRobot relay protocol")
    raw_poses = message.get("raw_poses")
    if not isinstance(raw_poses, dict):
        raise ValueError("XRobot relay packet has no raw_poses object")
    stream_id = str(message.get("stream_id", ""))
    if not stream_id:
        raise ValueError("XRobot relay packet has no stream_id")
    sequence = int(message.get("sequence", -1))
    if sequence < 0:
        raise ValueError("XRobot relay packet has invalid sequence")

    body = body_frame_from_raw_poses(
        {str(name): value for name, value in raw_poses.items()},
        timestamp_ns=int(message.get("timestamp_ns", 0)),
    )
    raw_controllers = message.get("controllers", {})
    controllers = {
        name: _controller_state(value)
        for name, value in raw_controllers.items()
        if name in ("left", "right")
    } if isinstance(raw_controllers, dict) else {}
    controller_frame = XRobotControllerFrame(
        controllers=controllers,
        timestamp_ns=int(message.get("controllers_timestamp_ns", 0)),
    )
    return body, controller_frame, stream_id, sequence


class UdpXRobotBodyReceiver:
    """Latest-frame UDP receiver compatible with ``XRobotBodyStreamer``.

    The socket is drained on every read so stale queued frames do not add
    latency. Missing packets return ``None``; callers can then keep their last
    GMR/RGMT reference exactly as they do for a paused local stream.
    """

    def __init__(
        self,
        *,
        bind_address: str = "0.0.0.0",
        port: int = 47001,
        source_host: str | None = None,
    ) -> None:
        self.bind_address = str(bind_address)
        self.port = int(port)
        self.source_host = str(source_host).strip() if source_host else None
        self._socket: socket.socket | None = None
        self._stream_id: str | None = None
        self._last_sequence = -1
        self._controller_frame = XRobotControllerFrame(controllers={}, timestamp_ns=0)
        self.received_packets = 0
        self.invalid_packets = 0
        self.rejected_packets = 0
        self.last_sender: tuple[str, int] | None = None

    def start(self) -> None:
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_address, self.port))
        sock.setblocking(False)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def read_body_frame(self) -> XRobotBodyFrame | None:
        if self._socket is None:
            raise RuntimeError("UdpXRobotBodyReceiver.start() must be called first")
        latest: XRobotBodyFrame | None = None
        while True:
            try:
                payload, sender = self._socket.recvfrom(65535)
            except BlockingIOError:
                break
            self.received_packets += 1
            if self.source_host is not None and sender[0] != self.source_host:
                self.rejected_packets += 1
                continue
            try:
                body, controller_frame, stream_id, sequence = decode_body_packet(payload)
            except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
                self.invalid_packets += 1
                continue
            if stream_id != self._stream_id:
                self._stream_id = stream_id
                self._last_sequence = -1
            if sequence <= self._last_sequence:
                continue
            self._last_sequence = sequence
            self._controller_frame = controller_frame
            self.last_sender = sender
            latest = body
        return latest

    def read_controller_frame(self) -> XRobotControllerFrame:
        return self._controller_frame


class XRobotFrameRelay:
    """Send local SDK frames to one UDP destination."""

    def __init__(self, *, target_host: str, target_port: int) -> None:
        self.target = (str(target_host), int(target_port))
        self.stream_id = uuid.uuid4().hex
        self.sequence = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(
        self,
        body_frame: XRobotBodyFrame,
        controller_frame: XRobotControllerFrame | None = None,
    ) -> int:
        payload = encode_body_packet(
            body_frame,
            controller_frame,
            sequence=self.sequence,
            stream_id=self.stream_id,
        )
        sent = self._socket.sendto(payload, self.target)
        self.sequence += 1
        return sent

    def close(self) -> None:
        self._socket.close()
