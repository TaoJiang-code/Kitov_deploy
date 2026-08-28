#!/usr/bin/env python3
"""Validate a Kitov deploy model bundle without starting XRobot or MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kitov_deploy.policy_runtime import (
    DEFAULT_MODEL_ROOT,
    ForwardKinematics,
    load_robot_policy_config,
    resolve_model_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Kitov policy ONNX bundle layout and metadata.")
    parser.add_argument("--robot", choices=["bumi", "g1", "unitree_g1"], default="bumi")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None, help="Override the robot model directory.")
    parser.add_argument("--check-fk", action="store_true", help="Also load the robot XML with MuJoCo and validate FK body names.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        robot_config = load_robot_policy_config(args.robot)
        bundle = resolve_model_bundle(args.robot, model_root=args.model_root, model_dir=args.model_dir)
    except Exception as exc:
        print(f"[check_policy_model] error: {exc}")
        return 1

    print(f"[check_policy_model] robot={robot_config.robot_name}")
    print(f"[check_policy_model] model_dir={bundle.model_dir}")
    print(f"[check_policy_model] policy_onnx={bundle.policy_onnx}")
    print(f"[check_policy_model] policy_meta={bundle.policy_meta}")
    print(f"[check_policy_model] backward_onnx={bundle.backward_onnx}")
    print(f"[check_policy_model] num_dof={bundle.metadata.get('num_dof')} z_dim={bundle.metadata.get('z_dim')}")
    print(f"[check_policy_model] actor_input_keys={bundle.metadata.get('actor_input_keys')}")
    print(f"[check_policy_model] actor_input_dims={bundle.metadata.get('actor_input_dims')}")
    print(f"[check_policy_model] actor_obs_dim={bundle.metadata.get('actor_obs_dim')}")

    model_joints = [str(name) for name in bundle.metadata.get("control_joint_names", [])]
    if model_joints and model_joints != robot_config.control_joint_names:
        raise SystemExit("metadata control_joint_names do not match deploy config")
    if int(bundle.metadata.get("num_dof", -1)) != robot_config.num_dof:
        raise SystemExit("metadata num_dof does not match deploy config")

    if args.check_fk:
        try:
            fk = ForwardKinematics(robot_config)
        except Exception as exc:
            print(f"[check_policy_model] fk error: {exc}")
            return 1
        print(f"[check_policy_model] fk_xml={fk.robot_config.xml_path}")
        print(f"[check_policy_model] fk_bodies={len(fk.body_ids)} extend={len(robot_config.extend_config)}")

    print("[check_policy_model] ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
