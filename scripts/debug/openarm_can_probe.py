#!/usr/bin/env python3
"""Probe OpenArm CAN motor state without enabling motors."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.hardware.openarm_can_bridge import (  # noqa: E402
    DEFAULT_OPENARM_HARDWARE_CONFIG,
    OpenArmCANBridge,
    load_openarm_hardware_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read OpenArm CAN motor state without enabling motors.")
    parser.add_argument("--hardware-config", type=Path, default=DEFAULT_OPENARM_HARDWARE_CONFIG)
    parser.add_argument("--hz", type=float, default=10.0, help="State refresh rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds. 0 means run until Ctrl-C.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Status print interval in seconds.")
    return parser.parse_args()


def _state_line(states: dict[str, Any]) -> str:
    parts = []
    for joint_name in sorted(states):
        state = states[joint_name]
        if state.velocity is None:
            parts.append(f"{joint_name}={state.position:.4f}")
        else:
            parts.append(f"{joint_name}={state.position:.4f}/{state.velocity:.4f}")
    return " ".join(parts)


def main() -> int:
    args = _parse_args()
    config = load_openarm_hardware_config(args.hardware_config)
    bridge = OpenArmCANBridge(config)

    stop = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("[openarm_can_probe] connecting without enabling motors")
    for bus in config.buses:
        if not bus.enabled:
            continue
        ids = ", ".join(f"{motor.send_can_id}->{motor.recv_can_id}" for motor in bus.motors if motor.enabled)
        print(f"[openarm_can_probe] bus side={bus.side} interface={bus.interface} can_fd={bus.can_fd} ids={ids}")
    bridge.connect()

    period_s = 1.0 / max(float(args.hz), 1e-6)
    print_interval = max(float(args.print_every), 0.05)
    start_time = time.monotonic()
    last_print = 0.0
    frames = 0

    try:
        while not stop:
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break
            states = bridge.read_state()
            frames += 1
            if now - last_print >= print_interval:
                print(f"[openarm_can_probe] t={now - start_time:.1f}s frames={frames} {_state_line(states)}")
                last_print = now
            elapsed = time.monotonic() - now
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    finally:
        print(f"[openarm_can_probe] stopped frames={frames}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
