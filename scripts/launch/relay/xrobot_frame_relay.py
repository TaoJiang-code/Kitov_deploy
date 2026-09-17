#!/usr/bin/env python3
"""Forward raw XRobot/PICO body frames from the PC to a remote BUMI host."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.xrobot_relay import XRobotFrameRelay
from kitov_deploy.xrobot_stream import XRobotBodyStreamer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward raw XRobot/PICO frames over UDP; GMR remains on the receiver."
    )
    parser.add_argument("--target-host", default="192.168.110.83", help="BUMI IP address.")
    parser.add_argument("--target-port", type=int, default=47001, help="UDP port on BUMI.")
    parser.add_argument("--hz", type=float, default=72.0, help="Maximum relay polling/send rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds; 0 means Ctrl-C.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Status interval in seconds.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    period_s = 1.0 / max(float(args.hz), 1e-6)
    print_interval = max(float(args.print_every), 0.05)
    stop = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    streamer = XRobotBodyStreamer()
    relay = XRobotFrameRelay(target_host=args.target_host, target_port=args.target_port)
    streamer.start()
    print(
        "[xrobot_frame_relay] started "
        f"target={args.target_host}:{args.target_port} hz={args.hz:.1f} "
        "payload=raw_unity_body_pose"
    )

    start_time = time.monotonic()
    last_loop = start_time
    last_print = start_time
    frames = 0
    missing = 0
    bytes_sent = 0
    try:
        while not stop:
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break

            frame = streamer.read_raw_body_frame()
            if frame is None:
                missing += 1
            else:
                controller_frame = streamer.read_controller_frame()
                bytes_sent += relay.send(frame, controller_frame)
                frames += 1

            elapsed = time.monotonic() - last_loop
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
            last_loop = time.monotonic()

            now = time.monotonic()
            if now - last_print >= print_interval:
                avg_hz = frames / max(now - start_time, 1e-6)
                print(
                    "[xrobot_frame_relay] "
                    f"t={now - start_time:.1f}s frames={frames} missing={missing} "
                    f"avg_hz={avg_hz:.1f} bytes={bytes_sent}"
                )
                last_print = now
    finally:
        streamer.close()
        relay.close()
        print(f"[xrobot_frame_relay] stopped frames={frames} missing={missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
