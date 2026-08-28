# Kitov Deploy

English | [中文](README_zh-CN.md)

Deployment runtime for models trained in:

```text
/home/unitree/robot_code/Kitov/Kitov
```

The first target is the existing BUMI model. The current implemented runtime
path is:

```text
PICO XRoboToolkit app
  -> XRoboToolkit PC Service
  -> xrobotoolkit_sdk
  -> Kitov_deploy XRobot reader
  -> GMR online retargeting
  -> G1 or BUMI qpos
  -> backward_encoder.onnx + policy ONNX
  -> q_target
  -> MuJoCo PD sim2sim viewer
```

This repository does not require changing directory into the GMR project at
runtime, and it no longer imports code from the external GMR repository. The
online retargeting implementation lives in this repository:

```text
kitov_deploy/gmr_online.py
configs/gmr/
configs/policy/
```

## Current Status

Implemented:

```text
scripts/debug/xrobot_probe.py
scripts/debug/xrobot_retarget.py
scripts/debug/openarm_can_probe.py
scripts/debug/replay_xrobot_frame.py
scripts/debug/check_policy_model.py
scripts/debug/replay_bfm_policy.py
scripts/xrobot_policy_infer.py
scripts/xrobot_openarm_control.py
scripts/run_bumi_policy_sim.sh
scripts/run_g1_policy_sim.sh
kitov_deploy/xrobot_stream.py
kitov_deploy/gmr_online.py
kitov_deploy/hardware/openarm_can_bridge.py
kitov_deploy/policy_runtime.py
kitov_deploy/mujoco_policy_sim.py
configs/gmr/xrobot_to_g1.json
configs/gmr/xrobot_to_bumi.json
configs/gmr/xrobot_to_openarm_v1.json
configs/hardware/openarm_v1.json
configs/policy/g1.json
configs/policy/bumi.json
```

Supported online retargeting targets:

```text
g1 / unitree_g1
bumi
openarm / openarm_v1
```

## Initialize Submodules

G1, BUMI, and OpenArm v1 robot XML / meshes come from the `Glush_Zoo` submodule.
The OpenArm v1 CAN SDK comes from the `third_party/openarm_can` submodule:

```bash
git submodule update --init --recursive
```

The current BUMI config uses:

```text
Glush_Zoo/bumi/mjcf/scene_21dof.xml
```

The current G1 config uses:

```text
Glush_Zoo/g1_description/mjcf/scene_29dof.xml
```

The current OpenArm v1 retargeting path uses:

```text
Glush_Zoo/openarm_v1/mjcf/scene.xml
configs/gmr/xrobot_to_openarm_v1.json
```

OpenArm v1 is currently wired only for online retargeting, not policy/model
inference.

OpenArm CAN SDK path:

```text
third_party/openarm_can
```

## Create Conda Environment

```bash
conda create -n Kitov_deploy python=3.10 -y
conda activate Kitov_deploy
```

Install runtime dependencies:

```bash
conda install -c conda-forge pybind11 libstdcxx-ng -y
pip install numpy scipy pyzmq
pip install mujoco mink daqp 'qpsolvers[proxqp]' loop-rate-limiters rich tqdm protobuf imageio onnxruntime
```

Verify core dependencies:

```bash
python - <<'PY'
import numpy, scipy, zmq
import mujoco, mink, qpsolvers, daqp, loop_rate_limiters, rich, imageio
import onnxruntime
print("Kitov_deploy deps ok")
PY
```

## XRobot Python SDK

`xrobotoolkit_sdk` must be installed inside the `Kitov_deploy` conda
environment. Verify it with:

```bash
python - <<'PY'
import xrobotoolkit_sdk as xrt
print("xrobotoolkit_sdk import ok")
print("init:", hasattr(xrt, "init"))
print("callback:", hasattr(xrt, "register_frame_callback"))
print("polling:", hasattr(xrt, "is_body_data_available"))
PY
```

## Install And Start XRoboToolkit PC Service

Prebuilt packages are stored in this repository:

```text
packages/xrobotoolkit_pc_service/
  XRoboToolkit_PC_Service_1.0.0_ubuntu_20.04_amd64.deb
  XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Install the package that matches the host Ubuntu version. On this machine
(`Ubuntu 20.04.6`), use:

```bash
cd /home/unitree/robot_code/Kitov/Kitov_deploy
sudo dpkg -i packages/xrobotoolkit_pc_service/XRoboToolkit_PC_Service_1.0.0_ubuntu_20.04_amd64.deb
```

For Ubuntu 22.04, install the 22.04 package instead:

```bash
sudo dpkg -i packages/xrobotoolkit_pc_service/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

If `dpkg` reports missing system dependencies:

```bash
sudo apt-get install -f
```

Start the service:

```bash
/opt/apps/roboticsservice/runService.sh
```

`runService.sh` starts `RoboticsServiceProcess` in the background and then
returns to the shell.

Check whether it is running:

```bash
pgrep -af RoboticsServiceProcess
```

Stop the service:

```bash
pkill -f RoboticsServiceProcess
```

## Run XRobot Probe

After the PC Service is running and the PICO app is connected to the PC IP:

```bash
cd /home/unitree/robot_code/Kitov/Kitov_deploy
conda activate Kitov_deploy
python scripts/debug/xrobot_probe.py --hz 50 --print-every 1
```

Expected healthy output:

```text
body_available=True
poses=24
body_hz around 50
Pelvis / Left_Foot / Right_Foot pose values update
A/B/X/Y button values respond
```

Force polling mode if callback mode is not usable:

```bash
python scripts/debug/xrobot_probe.py --mode polling --hz 50 --print-every 1
```

## Run Online GMR Retargeting

Retarget live PICO body data to BUMI qpos:

```bash
python scripts/debug/xrobot_retarget.py --robot bumi --hz 50 --quiet-gmr
```

Retarget live PICO body data to G1 qpos:

```bash
python scripts/debug/xrobot_retarget.py --robot g1 --hz 50 --quiet-gmr
```

Retarget live PICO body data to OpenArm v1 qpos:

```bash
python scripts/debug/xrobot_retarget.py --robot openarm_v1 --hz 50 --quiet-gmr
```

Run for a fixed test duration:

```bash
python scripts/debug/xrobot_retarget.py --robot bumi --duration 10 --print-every 1 --quiet-gmr
```

Print the full qpos vector:

```bash
python scripts/debug/xrobot_retarget.py --robot bumi --print-qpos all --quiet-gmr
```

Open the MuJoCo viewer:

```bash
python scripts/debug/xrobot_retarget.py --robot bumi --viewer --show-human --quiet-gmr
```

OpenArm v1 viewer:

```bash
python scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer --show-human --quiet-gmr
```

Draw only human axes and include all XRobot body joint names:

```bash
python scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer --show-human --show-all-human --human-axes-only --show-human-name --quiet-gmr
```

Tune the OpenArm v1 IK JSON against one saved XRobot frame:

```bash
python scripts/debug/tune_xrobot_ik_config.py \
  date/20260725_164701_673558_g1_frame003344.json \
  --robot openarm_v1 \
  --quiet-gmr
```

Use the `Human Display` tab in the tuning panel to choose which human frames are drawn. By default it only selects joints referenced by `ik_match_table1/2`.

In viewer mode, press `P` to save the latest PICO/XRobot body frame:

```text
recordings/xrobot_frames/*.json
```

The saved JSON includes raw SDK Unity poses, converted GMR poses, timestamp, and
the current retargeted qpos. Use `--save-dir` to choose a different directory.

Replay a saved frame through BUMI retargeting:

```bash
python scripts/debug/replay_xrobot_frame.py date/20260725_164701_673558_g1_frame003344.json --robot bumi --offset-to-ground --viewer --show-human --quiet-gmr
```

## OpenArm v1 Hardware Interface

OpenArm v1 hardware control currently stops at retargeted qpos. It does not use
policy/model inference:

```text
PICO / XRobot
  -> Kitov_deploy local GMR
  -> openarm_v1 qpos
  -> safety clamp / rate limit
  -> openarm_can
  -> SocketCAN
  -> Damiao motors
```

Hardware mapping lives in:

```text
configs/hardware/openarm_v1.json
```

It defines the CAN interface, motor type, send/receive CAN IDs, direction
`sign`, `zero_offset`, `kp/kd`, and max velocity for each MuJoCo joint. The
current default assumes right arm on `can0`, left arm on `can1`, motor IDs
`0x01..0x07`, and receive IDs `0x11..0x17` on each bus. Treat this only as a
starting point; real hardware needs per-joint direction and zero calibration.

Install OpenArm CAN from the submodule:

```bash
cd /home/unitree/robot_code/Kitov/Kitov_deploy/third_party/openarm_can
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build

cd python
pip install .
```

Configure SocketCAN:

```bash
openarm-can-cli -i can0 can_configure
openarm-can-cli -i can1 can_configure
```

Read motor state without enabling motors:

```bash
python scripts/debug/openarm_can_probe.py --hz 10 --print-every 1
```

Live XRobot -> GMR -> OpenArm hardware target dry-run. This only prints targets;
it does not import `openarm_can` or send CAN frames:

```bash
python scripts/xrobot_openarm_control.py --hz 50 --quiet-gmr --print-targets head
```

Dry-run with MuJoCo viewer. This viewer shows the command qpos after
`sign / zero_offset / hardware_lower / hardware_upper / max_velocity_rad_s`,
not the raw GMR IK qpos:

```bash
python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --viewer \
  --show-human \
  --show-all-human \
  --human-axes-only \
  --show-human-name
```

`scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer` remains a pure
GMR retargeting check. Use `xrobot_openarm_control.py --viewer` to see the
targets that would be sent to hardware.

Real hardware sending requires explicit `--send --enable-motors`:

```bash
python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --send \
  --enable-motors
```

On exit, the script disables motors by default. If PICO/XRobot frames stop
updating longer than `watchdog_timeout_s`, it also disables motors and exits.

## Model Inference

Model files are not committed to this repository. Copy each exported model bundle
manually:

```text
models/bumi/exported/
  FBcprAuxModel.onnx
  FBcprAuxModel.meta.json
  backward_encoder.onnx

models/g1/exported/
  FBcprAuxModel.onnx
  FBcprAuxModel.meta.json
  backward_encoder.onnx
```

Check a copied BUMI bundle:

```bash
python scripts/debug/check_policy_model.py --robot bumi --check-fk
```

Run live XRobot -> GMR -> policy inference without a MuJoCo window:

```bash
python scripts/xrobot_policy_infer.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --hz 50 \
  --offset-to-ground \
  --quiet-gmr
```

Run live XRobot -> GMR -> policy -> MuJoCo sim2sim viewer:

```bash
python scripts/xrobot_policy_infer.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --hz 50 \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human
```

This opens two windows:

```text
policy MuJoCo viewer: policy q_target applied through PD torque control, with floor/light injected
GMR viewer: live retargeted reference qpos, with human targets when --show-human is set
```

The GMR viewer is launched in a separate child process to avoid MuJoCo/GLFW
segfaults from two passive viewers in one Python process. It is still started
and stopped from the same terminal command.

The short launcher is equivalent and opens both windows:

```bash
scripts/run_bumi_policy_sim.sh
```

For G1:

```bash
python scripts/debug/check_policy_model.py --robot g1 --check-fk
python scripts/xrobot_policy_infer.py \
  --robot g1 \
  --model-dir models/g1/kitov_fb_g1 \
  --hz 50 \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human \
  --elastic-band \
  --elastic-length 1.5

scripts/run_g1_policy_sim.sh
```

The inference runtime follows the Kitov training export metadata:

```text
backward_encoder.onnx:
  state + last_action + privileged_state -> z

policy ONNX:
  actor_obs = state + last_action/history_actor as requested by metadata + z
  action -> normalized action -> q_target
```

In `--viewer` mode, `q_target` is applied to the same robot XML through PD
torque control. The PD gains and torque limits are stored in
`configs/policy/g1.json` and `configs/policy/bumi.json`. These files also
store the training-side actuator armature and frictionloss values that are
patched into the MuJoCo model at runtime.

For G1 sim bring-up, `--elastic-band` applies the same soft root support style
used by UFO-Deploy. The default spring constants match UFO. `--elastic-length`
can be tuned; `1.5` is a practical starting value for G1 replay/viewer tests.

For policy sim diagnosis, switch the MuJoCo command source:

```bash
--control-source policy     # default, command policy q_target
--control-source reference  # command GMR reference dof_pos directly
--control-source hold       # hold the current joints
```

Replay a BFM motion from the training dataset through the deploy-side ONNX + PD
diagnostic path, without PICO/GMR online streaming:

```bash
python scripts/debug/replay_bfm_policy.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --data-path /home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/bumi/bumi_lafan_full.pkl \
  --motion-index 0 \
  --max-frames 600
```

For G1, do not use motion index `0` as a stability test because it is a
fall/get-up clip. If no `--motion-index` or `--motion-key` is provided, the
script skips obvious fall/get-up/lie clips and starts from the first ordinary
motion:

```bash
python scripts/debug/replay_bfm_policy.py \
  --robot g1 \
  --model-dir models/g1/kitov_fb_g1 \
  --data-path /home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/g1/lafan_29dof.pkl \
  --max-frames 600 \
  --elastic-band \
  --elastic-length 1.5
```

List available motion keys:

```bash
python scripts/debug/replay_bfm_policy.py --robot g1 --list-motions
```

This script first converts the expert pkl motion into `z_seq` through
`backward_encoder.onnx`, then runs `policy ONNX -> q_target` frame by frame.

Use `--no-floor` only if you want to load the original robot XML without the
viewer floor/light injection. Press `Ctrl-C` in the terminal to close both
viewers and stop the XRobot stream.
