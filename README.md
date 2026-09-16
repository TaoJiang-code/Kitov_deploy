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

BUMI also supports an RGMT policy path:

```text
PICO/XRobot -> GMR online retargeting -> BUMI qpos
  -> RGMT policy.onnx
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
scripts/debug/openarm_hardware_tuner.py
scripts/debug/replay_xrobot_frame.py
scripts/debug/check_policy_model.py
scripts/debug/replay_bfm_policy.py
scripts/xrobot_policy_infer.py
scripts/xrobot_rgmt_policy_infer.py
scripts/xrobot_bumi_rgmt_policy_real.py
scripts/xrobot_openarm_control.py
scripts/run_openarm_teleop.sh
scripts/run_bumi_policy_sim.sh
scripts/run_bumi_rgmt_policy_sim.sh
scripts/run_bumi_policy_real.sh
scripts/run_g1_policy_sim.sh
kitov_deploy/xrobot_stream.py
kitov_deploy/gmr_online.py
kitov_deploy/hardware/openarm_can_bridge.py
kitov_deploy/hardware/bumi_noetix_bridge.py
ee_body/
kitov_deploy/policy_runtime.py
kitov_deploy/mujoco_policy_sim.py
configs/gmr/xrobot_to_g1.json
configs/gmr/xrobot_to_bumi.json
configs/gmr/xrobot_to_openarm_v1.json
configs/hardware/openarm_v1.json
configs/hardware/bumi_noetix.json
configs/policy/g1.json
configs/policy/bumi.json
configs/policy/bumi_rgmt.json
```

Supported online retargeting targets:

```text
g1 / unitree_g1
bumi
openarm / openarm_v1
```

## Initialize Submodules

G1, BUMI, and OpenArm v1 robot XML / meshes come from the `Glush_Zoo` submodule.
The OpenArm v1 CAN SDK comes from the `third_party/openarm_can` submodule. The
Noetix SDK for BUMI hardware lives in the `third_party/noetix_sdk_bumi`
submodule:

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

Noetix BUMI SDK path:

```text
third_party/noetix_sdk_bumi
```

BUMI real-hardware control does not use `rl_real_g1`. `rl_real_g1
<YOUR_NETWORK_INTERFACE>` is the G1 / Unitree DDS entrypoint, where the
argument is the local network interface. In `rl_sar`, BUMI uses
`rl_real_bumi`; its first argument is a CycloneDDS XML path. If no argument is
provided, it tries the Noetix SDK default `config/dds.xml`.

```bash
./cmake_build/bin/rl_real_bumi
# or explicitly pass a DDS config
./cmake_build/bin/rl_real_bumi /path/to/dds.xml
```

If `cmake_build/bin` contains `rl_real_g1` but not `rl_real_bumi`, the BUMI
real target was not built. Do not use `rl_real_g1` as a substitute.

## Create uv Environment

Recommended setup command. Run it and choose from the menu:

```bash
scripts/tool/setup_env.sh
```

Menu choices:

```text
1) all          ONNX Runtime + PyTorch
2) onnxruntime  ONNX Runtime only
3) torch        PyTorch only
4) skip         uv sync only
```

For non-interactive setup, use `KITOV_INSTALL_TARGET`:

```bash
KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=all scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=skip scripts/tool/setup_env.sh
```

The default `KITOV_ONNXRUNTIME_MODE=auto` installs the regular CPU ONNX Runtime
on x86. On Jetson, it installs `onnxruntime-gpu==1.23.0` from the Jetson AI Lab
JetPack 6 / cu126 index. For other JetPack/L4T versions, pass a matching wheel:

```bash
KITOV_JETSON_ONNXRUNTIME_WHEEL=/path/to/onnxruntime_gpu-xxx-linux_aarch64.whl \
  KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
```

The JetPack 6 default version can also be overridden:

```bash
KITOV_JETSON_ONNXRUNTIME_VERSION=1.23.0 KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
```

The default `KITOV_TORCH_MODE=auto` checks `nvidia-smi` on x86 and installs a
CUDA PyTorch wheel from `cu128` or `cu126` based on the driver capability.
Jetson is `aarch64` and cannot use the x86 `cu128/cu126` wheels; pass the
matching NVIDIA Jetson wheel for that JetPack/L4T version:

```bash
KITOV_JETSON_TORCH_WHEEL=/path/to/torch-xxx-linux_aarch64.whl \
  KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
```

Manual mode overrides:

```bash
KITOV_ONNXRUNTIME_MODE=cpu KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
KITOV_ONNXRUNTIME_MODE=jetson-gpu KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
KITOV_TORCH_MODE=cu128 KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_TORCH_MODE=cu126 KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_TORCH_MODE=cpu KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_TORCH_MODE=skip KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
```

If `.venv` already exists, the script reuses it by default. To recreate it:

```bash
KITOV_RECREATE_VENV=1 scripts/tool/setup_env.sh
```

Manual step-by-step setup:

```bash
uv venv --python 3.10
uv sync --inexact
```

If `uv` is not installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Python dependencies are managed by `pyproject.toml`. `--inexact` preserves
local bindings manually installed into `.venv`. `uv` only manages Python
packages; `XRoboToolkit PC Service`, `xrobotoolkit_sdk`, and `openarm_can`
service/C++/pybind components still need to be installed into the active `.venv`.

Verify core dependencies:

```bash
uv run python - <<'PY'
import numpy, scipy, zmq
import mujoco, mink, qpsolvers, daqp, loop_rate_limiters, rich, imageio
import onnxruntime
print("Kitov_deploy deps ok")
PY
```

## XRobot Python SDK

`xrobotoolkit_sdk` must be installed inside the current `.venv`. An SDK
installed in the old conda environment is not visible from the uv environment.

On a new machine, prepare both repositories inside
`workspace/xrobot_toolkit/` under the repository root. This directory is ignored
by Git. Run each block below from the Kitov_deploy repository root:

```bash
mkdir -p workspace/xrobot_toolkit
cd workspace/xrobot_toolkit

git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
```

Build the XRoboToolkit C++ SDK:

```bash
cd workspace/xrobot_toolkit/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
```

Copy the C++ SDK artifacts into the Python binding project:

```bash
cd workspace/xrobot_toolkit/XRoboToolkit-PC-Service-Pybind
mkdir -p lib include

cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
```

Install the binding into the Kitov_deploy uv environment:

```bash
uv pip install workspace/xrobot_toolkit/XRoboToolkit-PC-Service-Pybind
```

Verify it with:

```bash
uv run python - <<'PY'
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
uv run python scripts/debug/xrobot_probe.py --hz 50 --print-every 1
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
uv run python scripts/debug/xrobot_probe.py --mode polling --hz 50 --print-every 1
```

## Run Online GMR Retargeting

Retarget live PICO body data to BUMI qpos:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --hz 50 --quiet-gmr
```

Retarget live PICO body data to G1 qpos:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot g1 --hz 50 --quiet-gmr
```

Retarget live PICO body data to OpenArm v1 qpos:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot openarm_v1 --hz 50 --quiet-gmr
```

Run for a fixed test duration:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --duration 10 --print-every 1 --quiet-gmr
```

Print the full qpos vector:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --print-qpos all --quiet-gmr
```

Open the MuJoCo viewer:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --viewer --show-human --quiet-gmr
```

OpenArm v1 viewer:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer --show-human --quiet-gmr
```

Draw only human axes and include all XRobot body joint names:

```bash
uv run python scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer --show-human --show-all-human --human-axes-only --show-human-name --quiet-gmr
```

Tune the OpenArm v1 IK JSON against one saved XRobot frame:

```bash
uv run python scripts/debug/tune_xrobot_ik_config.py \
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
uv run python scripts/debug/replay_xrobot_frame.py date/20260725_164701_673558_g1_frame003344.json --robot bumi --offset-to-ground --viewer --show-human --quiet-gmr
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
`sign`, `zero_offset`, `kp/kd`, and max velocity for each MuJoCo joint.
If `hardware_lower/hardware_upper` is `null`, the XML joint range is used and
converted into hardware coordinates through `sign/zero_offset`; explicit JSON
values override the XML range. `recv_timeout_us` is the normal state receive
timeout; `enable_recv_timeout_us` is the receive timeout after enable / disable,
currently 500ms to match the OpenArm CLI behavior. The current default follows
the observed hardware wiring: right arm on `can1`, left arm on `can0`, motor IDs `0x01..0x07`, and receive IDs
`0x11..0x17` on each bus. Treat this only as a starting point; real hardware
needs per-joint direction and zero calibration.

End-effector bodies live in a separate directory:

```text
ee_body/
  config/openarm_v1_dm_gripper.json
  drivers/openarm_can_gripper.py
```

`configs/hardware/openarm_v1.json` keeps only one end-effector selector. It
currently selects the OpenArm DM gripper:

```json
"ee_body": "openarm_v1_dm_gripper"
```

Use `none` when no end-effector is attached:

```json
"ee_body": "none"
```

This loads `ee_body/config/openarm_v1_dm_gripper.json` and uses the implementation in
`ee_body/drivers/openarm_can_gripper.py`. End-effector position sending, state reading,
zeroing, `kp/kd`, speed, and torque limits are maintained under `ee_body/`. Arm
7DoF mapping remains in `configs/hardware/openarm_v1.json`.
If an end-effector plugin exposes a tuner panel,
`scripts/debug/openarm_hardware_tuner.py` mounts it automatically.

For the OpenArm DM gripper config, `open_position` is the open target,
`close_position` is the mechanical close limit, and `safe_close_position` is the
deepest target used by normal trigger control. `close_speed_rad_s` and
`close_torque_pu` are intentionally lower than the open values to avoid crushing
objects. If `torque_stop_threshold` is set, closing stops when feedback torque
crosses that threshold. The default `null` leaves torque-threshold stopping off
until the feedback units are verified, while still using low torque and stall
detection. `stall_velocity_threshold` and `stall_hold_time_s` detect contact
when the gripper is still commanded closed but velocity stays near zero; the
driver then holds the current gripper position until the trigger is released.

Install system dependencies before building OpenArm CAN. The CMake error
`Could not find CLI11` means `libcli11-dev` is missing:

```bash
sudo apt update
sudo apt install -y cmake build-essential libcli11-dev can-utils
```

If `libcli11-dev` is not available from the current Ubuntu package source,
install CLI11 from source first:

```bash
mkdir -p workspace/deps
wget -O workspace/deps/CLI11.zip https://github.com/CLIUtils/CLI11/archive/refs/heads/main.zip
unzip -q workspace/deps/CLI11.zip -d workspace/deps
mv workspace/deps/CLI11-main workspace/deps/CLI11

cmake -S workspace/deps/CLI11 -B workspace/deps/CLI11/build \
  -DCLI11_BUILD_DOCS=OFF \
  -DCLI11_BUILD_EXAMPLES=OFF \
  -DCLI11_BUILD_TESTS=OFF
sudo cmake --install workspace/deps/CLI11/build
```

Then build and install the OpenArm CAN submodule:

```bash
cd third_party/openarm_can
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build

cd python
uv pip install .
```

Configure SocketCAN:

```bash
openarm-can-cli -i can0 can_configure
openarm-can-cli -i can1 can_configure
```

The recommended OpenArm hardware teleop entrypoint checks and configures
`can0/can1`, makes sure `XRoboToolkit PC Service` is running, then starts
realtime control. `Ctrl+C` stops the control process and shuts down
`RoboticsServiceProcess`:

```bash
./scripts/run_openarm_teleop.sh
```

By default this is equivalent to:

```bash
uv run python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --send \
  --enable-motors \
  --enable-ee-control \
  --viewer \
  --show-human
```

Extra arguments are appended to the Python command:

```bash
./scripts/run_openarm_teleop.sh --print-targets head
```

Optional environment variables:

```bash
KITOV_HZ=50
KITOV_OPENARM_CAN_INTERFACES="can0 can1"
KITOV_OPENARM_CAN_BITRATE=1000000
KITOV_OPENARM_CAN_DBITRATE=5000000
KITOV_XROBOT_SERVICE_SCRIPT=/opt/apps/roboticsservice/runService.sh
KITOV_STOP_ROBOTICS_SERVICE_ON_EXIT=1
```

Read motor state without enabling motors:

```bash
uv run python scripts/debug/openarm_can_probe.py --hz 10 --print-every 1
```

Live XRobot -> GMR -> OpenArm hardware target dry-run. This only prints targets;
it does not import `openarm_can` or send CAN frames:

```bash
uv run python scripts/xrobot_openarm_control.py --hz 50 --quiet-gmr --print-targets head
```

Dry-run with MuJoCo viewer. This viewer shows the command qpos after
`sign / zero_offset / hardware_lower / hardware_upper / max_velocity_rad_s`,
not the raw GMR IK qpos:

```bash
uv run python scripts/xrobot_openarm_control.py \
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
uv run python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --send \
  --enable-motors
```

If `configs/hardware/openarm_v1.json` selects an end-effector, for example
`"ee_body": "openarm_v1_dm_gripper"`, add `--enable-ee-control` to drive the
grippers from PICO triggers:

```bash
uv run python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --send \
  --enable-motors \
  --enable-ee-control
```

The left trigger controls `left_gripper`; the right trigger controls
`right_gripper`. Released means open, pressed means closing toward
`safe_close_position`, using the low-speed, low-torque, contact-hold logic from
the end-effector config.

The script connects to CAN and then calls `enable_all` immediately. It then waits
briefly for stable current motor positions and only sends the MIT hold target
after the startup hold target passes `hardware_lower/hardware_upper` or XML joint
range validation. Startup readings are first folded by `2*pi` into the valid
range, so boundary-like values such as `-12.4rad` can map back into the current
joint range. If the folded value is still invalid, it is rejected. By default,
folded startup readings whose absolute value exceeds `--startup-position-abs-limit
6.283` are also rejected. Before the first XRobot body frame arrives, it keeps
sending the current motor positions as the hold target. During runtime, if
PICO/XRobot frames stop updating, the script keeps sending the last target so
the arm holds the current posture. On exit, it disables motors by default.

Standalone hardware tuning panel:

```bash
uv run python scripts/debug/openarm_hardware_tuner.py --hz 50
```

This panel does not use XRobot/GMR. It only tests
`configs/hardware/openarm_v1.json -> openarm_can`. The default is
`--slider-space sim`: sliders are MuJoCo joint angles,
then targets are converted through `sign/zero_offset` before being sent to
hardware. This matches the realtime control mapping path and is the right mode
for checking whether MuJoCo commands and the real arm move in the same direction.
Each joint row shows current motor feedback, feedback converted back into MuJoCo
coordinates, the target slider, the last sent target, error, and enabled state.
`Enable All` only enables motors, `Hold Current` copies feedback into the target
sliders, `Send Once` sends one `max_velocity_rad_s` limited target step, and
`Auto Send` continuously sends slider targets. `Flip Sign` flips that joint's
`sign` and writes it back to JSON. `Set Motor Zero All` calls the motor hardware
zero command; `Save JSON Zero Offset` only writes current feedback into this
repository's `zero_offset` fields and does not change motor-side zero.

If `configs/hardware/openarm_v1.json` selects an end effector, the bottom of the
panel automatically shows that plugin's controls. The current OpenArm DM gripper
plugin shows actual position, velocity, torque, and target sliders, and lets you
edit `sign`, `safe_close_position`, `open_speed_rad_s/open_torque_pu`,
`close_speed_rad_s/close_torque_pu`, and `torque_stop_threshold`. `Open` opens
the gripper, `Safe Close` closes to `safe_close_position` with the low-speed,
low-torque profile, and `Close Limit` sends the mechanical close limit.
`Flip Sign` changes the runtime direction first; `Save EE JSON` persists these
end-effector fields to `ee_body/config/openarm_v1_dm_gripper.json`.

For low-level raw motor-angle tests:

```bash
uv run python scripts/debug/openarm_hardware_tuner.py --hz 50 --slider-space hardware
```

## Model Inference

Model files are not committed to this repository. Copy exported files manually.

BFM zero / Kitov ONNX bundle:

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

BUMI RGMT ONNX:

```text
models/bumi/rgmt/
  policy.onnx
```

The RGMT deploy config is:

```text
configs/policy/bumi_rgmt.json
```

If both `policy.onnx` and `policy.pt` exist, `policy.onnx` is used first.
`policy.pt` is still supported as a fallback, but requires PyTorch in the active
uv environment.

Check a copied BUMI bundle:

```bash
uv run python scripts/debug/check_policy_model.py --robot bumi --check-fk
```

Run live XRobot -> GMR -> policy inference without a MuJoCo window:

```bash
uv run python scripts/xrobot_policy_infer.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --hz 50 \
  --offset-to-ground \
  --quiet-gmr
```

Run live XRobot -> GMR -> policy -> MuJoCo sim2sim viewer:

```bash
uv run python scripts/xrobot_policy_infer.py \
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

In `--viewer` mode, MuJoCo policy control starts in damping. Terminal and policy
viewer keys use the same mapping:

```text
p / P: damping
0:     reset joints to zero, then stay in damping
1:     enable policy control
```

The short launcher is equivalent and opens both windows:

```bash
scripts/run_bumi_policy_sim.sh
```

BUMI RGMT uses a separate policy path and does not use `backward_encoder.onnx`:

```bash
uv run python scripts/xrobot_rgmt_policy_infer.py \
  --model-dir models/bumi/rgmt \
  --hz 50 \
  --offset-to-ground \
  --quiet-gmr \
  --viewer \
  --gmr-viewer \
  --show-human
```

Short launcher:

```bash
scripts/run_bumi_rgmt_policy_sim.sh
```

BUMI RGMT real-hardware entrypoint through the Noetix SDK:

```bash
scripts/run_bumi_policy_real.sh
```

This launcher connects to `third_party/noetix_sdk_bumi` and sends motor
commands. It starts in damping mode:

```text
p / P: return to damping
0:     rate-limited joint-zero command
1:     enter RGMT policy only from zero mode; ignored from damping
```

Hardware parameters live in:

```text
configs/hardware/bumi_noetix.json
```

`max_velocity_rad_s` limits joint target slew rate. `policy_kp/policy_kd` are
used under policy control, and `zero_kp/zero_kd` are used for the `0` command.

If fresh XRobot body frames pause briefly while running, the real entry keeps
using the last PICO/GMR reference already stored in the RGMT buffer and lets the
policy hold its output. If `1` is pressed before enough RGMT reference frames
have been collected, the real entry keeps holding the `0` reset target. It only
stays in damping when no reset target exists and a valid policy step is not
available.

Live RGMT delays the policy reference center by `rgmt_command_window_after`
frames by default. With the current config this is 10 frames, about 0.2s at
50Hz. This lets the future command window use real received PICO/GMR references
instead of repeating the newest frame. To disable it:

```bash
scripts/run_bumi_rgmt_policy_sim.sh --reference-delay-frames 0
```

The RGMT inputs follow the training-side policy signature:

```text
rgmt_policy:         projected_gravity + base_ang_vel + dof_pos_rel + dof_vel + last_action
rgmt_state_history:  last 10 state_obs frames
rgmt_action_history: last 10 action frames
rgmt_command:        21-frame GMR reference window with anchor velocity, gravity direction, and reference joint pos
```

For G1:

```bash
uv run python scripts/debug/check_policy_model.py --robot g1 --check-fk
uv run python scripts/xrobot_policy_infer.py \
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
uv run python scripts/debug/replay_bfm_policy.py \
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
uv run python scripts/debug/replay_bfm_policy.py \
  --robot g1 \
  --model-dir models/g1/kitov_fb_g1 \
  --data-path /home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/g1/lafan_29dof.pkl \
  --max-frames 600 \
  --elastic-band \
  --elastic-length 1.5
```

List available motion keys:

```bash
uv run python scripts/debug/replay_bfm_policy.py --robot g1 --list-motions
```

This script first converts the expert pkl motion into `z_seq` through
`backward_encoder.onnx`, then runs `policy ONNX -> q_target` frame by frame.

Use `--no-floor` only if you want to load the original robot XML without the
viewer floor/light injection. Press `Ctrl-C` in the terminal to close both
viewers and stop the XRobot stream.
