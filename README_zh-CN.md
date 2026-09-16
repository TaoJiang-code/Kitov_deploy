# Kitov Deploy

[English](README.md) | 中文

本仓库用于部署这里训练出来的模型：

```text
/home/unitree/robot_code/Kitov/Kitov
```

第一版目标是已有的 BUMI 模型。当前已经实现的运行链路是：

```text
PICO XRoboToolkit app
  -> XRoboToolkit PC Service
  -> xrobotoolkit_sdk
  -> Kitov_deploy XRobot 读取器
  -> GMR 在线重定向
  -> G1 或 BUMI qpos
  -> backward_encoder.onnx + policy ONNX
  -> q_target
  -> MuJoCo PD sim2sim viewer
```

BUMI 还额外支持 RGMT 模型链路：

```text
PICO/XRobot -> GMR 在线重定向 -> BUMI qpos
  -> RGMT policy.onnx
  -> q_target
  -> MuJoCo PD sim2sim viewer
```

运行时不需要再切到 GMR 目录，也不再 import 外部 GMR 仓库代码。在线重定向逻辑已经放在本仓库：

```text
kitov_deploy/gmr_online.py
configs/gmr/
configs/policy/
```

## 当前状态

已经实现：

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

在线重定向目前支持：

```text
g1 / unitree_g1
bumi
openarm / openarm_v1
```

## 初始化 Submodule

G1、BUMI 和 OpenArm v1 的机器人 XML / mesh 来自本仓库的 `Glush_Zoo` submodule。
OpenArm v1 的 CAN SDK 来自 `third_party/openarm_can` submodule。BUMI 实机相关的
Noetix SDK 放在 `third_party/noetix_sdk_bumi` submodule：

```bash
git submodule update --init --recursive
```

当前 BUMI 配置使用：

```text
Glush_Zoo/bumi/mjcf/scene_21dof.xml
```

当前 G1 配置使用：

```text
Glush_Zoo/g1_description/mjcf/scene_29dof.xml
```

当前 OpenArm v1 重定向使用：

```text
Glush_Zoo/openarm_v1/mjcf/scene.xml
configs/gmr/xrobot_to_openarm_v1.json
```

OpenArm v1 目前只接在线重定向，不接 policy/model 推理。

OpenArm CAN SDK 位置：

```text
third_party/openarm_can
```

Noetix BUMI SDK 位置：

```text
third_party/noetix_sdk_bumi
```

BUMI 真机控制不使用 `rl_real_g1`。`rl_real_g1 <YOUR_NETWORK_INTERFACE>` 是
G1 / Unitree DDS 入口，参数是本机网卡名。BUMI 在 `rl_sar` 里对应的是
`rl_real_bumi`，并且它的第一个参数是 CycloneDDS XML 配置路径；不传参数时会尝试使用
Noetix SDK 自带的 `config/dds.xml`。

```bash
./cmake_build/bin/rl_real_bumi
# 或显式指定 DDS 配置
./cmake_build/bin/rl_real_bumi /path/to/dds.xml
```

如果 `cmake_build/bin` 里只有 `rl_real_g1`、没有 `rl_real_bumi`，说明 BUMI 真机
target 没有被编译出来，不应该改用 `rl_real_g1` 顶上。

## 创建 uv 环境

推荐直接用脚本创建 uv 环境。直接回车运行后按菜单选择：

```bash
scripts/tool/setup_env.sh
```

菜单含义：

```text
1) all          ONNX Runtime + PyTorch
2) onnxruntime  只装/替换 ONNX Runtime
3) torch        只装 PyTorch
4) skip         只 uv sync，不额外装 ORT/PyTorch
```

随后脚本还会询问 XRobot 相关安装：

```text
1) skip        不安装 XRobot SDK / PC Service
2) sdk         编译并安装 xrobotoolkit_sdk 到当前 .venv
3) service     安装 XRoboToolkit PC Service，优先 deb，不匹配则源码编译
4) all         SDK + PC Service
```

非交互场景可以用环境变量。只想处理 Python 运行时依赖时，把 XRobot 步骤设为 `skip`：

```bash
KITOV_INSTALL_TARGET=onnxruntime KITOV_XROBOT_SETUP=skip scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=torch KITOV_XROBOT_SETUP=skip scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=all KITOV_XROBOT_SETUP=skip scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=skip scripts/tool/setup_env.sh

KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=sdk scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=service scripts/tool/setup_env.sh
KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=all scripts/tool/setup_env.sh
```

默认 `KITOV_ONNXRUNTIME_MODE=auto`。x86 会装普通 CPU ONNX Runtime；Jetson
会优先从 Jetson AI Lab 的 JetPack 6 / cu126 索引安装 `onnxruntime-gpu==1.23.0`。
这个 Jetson wheel 使用 NumPy 1.x ABI，所以项目依赖默认固定为 `numpy<2`。
如果不是 JetPack 6，手动指定匹配当前 JetPack/L4T 的 wheel：

```bash
KITOV_JETSON_ONNXRUNTIME_WHEEL=/path/to/onnxruntime_gpu-xxx-linux_aarch64.whl \
  KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
```

JetPack 6 默认版本也可以覆盖：

```bash
KITOV_JETSON_ONNXRUNTIME_VERSION=1.23.0 KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
```

默认 `KITOV_TORCH_MODE=auto`。x86 机器会检查 `nvidia-smi`，并按驱动支持的
CUDA capability 自动选择 `cu128` 或 `cu126` PyTorch wheel。Jetson 是 `aarch64`，
不能使用 x86 的 `cu128/cu126` wheel，需要指定匹配 JetPack/L4T 的 NVIDIA Jetson wheel：

```bash
KITOV_JETSON_TORCH_WHEEL=/path/to/torch-xxx-linux_aarch64.whl \
  KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
```

也可以手动指定：

```bash
KITOV_ONNXRUNTIME_MODE=cpu KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
KITOV_ONNXRUNTIME_MODE=jetson-gpu KITOV_INSTALL_TARGET=onnxruntime scripts/tool/setup_env.sh
KITOV_TORCH_MODE=cu128 KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_TORCH_MODE=cu126 KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_TORCH_MODE=cpu KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
KITOV_TORCH_MODE=skip KITOV_INSTALL_TARGET=torch scripts/tool/setup_env.sh
```

如果 `.venv` 已存在，脚本默认复用它；需要重建时再显式指定：

```bash
KITOV_RECREATE_VENV=1 scripts/tool/setup_env.sh
```

分步创建环境：

```bash
uv venv --python 3.10
uv sync --inexact
```

如果当前机器还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Python 依赖由 `pyproject.toml` 管理。这里用 `--inexact` 是为了保留手动安装进
`.venv` 的本地 binding。`uv` 只负责 Python 包；`XRoboToolkit PC
Service`、`xrobotoolkit_sdk`、`openarm_can` 这种服务端或 C++/pybind 绑定仍然要装到
当前 `.venv` 里。

验证核心依赖：

```bash
uv run python - <<'PY'
import numpy, scipy, zmq
import mujoco, mink, qpsolvers, daqp, loop_rate_limiters, rich, imageio
import onnxruntime
print("Kitov_deploy deps ok")
PY
```

## XRobot Python SDK

`xrobotoolkit_sdk` 必须安装在当前 `.venv` 里。之前装在 conda 环境里的 SDK 不会自动进入
uv 环境。

推荐直接通过环境脚本安装，选择第二个菜单里的 `sdk` 或 `all`：

```bash
scripts/tool/setup_env.sh
```

脚本会把临时仓库放在 `workspace/xrobot_toolkit/`，这个目录已加入 `.gitignore`。它会自动：

```text
clone XRoboToolkit-PC-Service-Pybind
clone XRoboToolkit-PC-Service
build RoboticsService/PXREARobotSDK
复制 PXREARobotSDK.h / nlohmann / libPXREARobotSDK.so 到 pybind 项目
uv pip install workspace/xrobot_toolkit/XRoboToolkit-PC-Service-Pybind
```

Jetson/aarch64 会自动使用 XRoboToolkit-PC-Service 的 `orin` 分支，并 clone 到独立目录
`workspace/xrobot_toolkit/XRoboToolkit-PC-Service-orin`，避免和 x86/main 分支工作区混用。
如需覆盖分支：

```bash
KITOV_XROBOT_SERVICE_REF=<branch-or-tag> KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=sdk scripts/tool/setup_env.sh
```

Jetson/aarch64 上如果 XRoboToolkit 仓库自带的 grpc include 缺
`google/protobuf/runtime_version.h`，脚本会自动下载 protobuf `v27.2` 源码包，并把对应
C++ headers 补到当前 XRoboToolkit-PC-Service 工作区的 bundled include
目录后再编译。protobuf 27.x 还依赖 Abseil headers，脚本也会自动补
`abseil-cpp 20240116.2`。如果 bundled include 继续缺 `grpcpp/...`，脚本会自动补
`grpc v1.64.0` 的 public headers。

非交互安装只装 SDK：

```bash
KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=sdk scripts/tool/setup_env.sh
```

验证方式：

```bash
uv run python - <<'PY'
import xrobotoolkit_sdk as xrt
print("xrobotoolkit_sdk import ok")
print("init:", hasattr(xrt, "init"))
print("callback:", hasattr(xrt, "register_frame_callback"))
print("polling:", hasattr(xrt, "is_body_data_available"))
PY
```

## 安装并启动 XRoboToolkit PC Service

本仓库已经放好了 XRoboToolkit PC Service 安装包：

```text
packages/xrobotoolkit_pc_service/
  XRoboToolkit_PC_Service_1.0.0_ubuntu_20.04_amd64.deb
  XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

推荐通过环境脚本自动安装，选择第二个菜单里的 `service` 或 `all`：

```bash
scripts/tool/setup_env.sh
```

脚本会优先检测是否有匹配当前系统的 `.deb`。仓库里现有包覆盖 `amd64` 的 Ubuntu
`20.04` / `22.04`；如果匹配成功就安装 `.deb`，并在 `dpkg` 提示依赖缺失时自动执行
`sudo apt-get install -f -y` 后重试。

如果没有匹配 `.deb`，例如 Jetson `aarch64`，脚本会自动走源码安装：在
`workspace/xrobot_toolkit/XRoboToolkit-PC-Service` 下 clone/复用源码，执行
`RoboticsService/qt-gcc.sh` 编译，然后把 `RoboticsService/bin` 安装到
`/opt/apps/roboticsservice`。源码编译依赖 Qt；如果当前机器没有 XRoboToolkit 需要的 Qt，
脚本会在编译阶段报错，需要先装 Qt 后重跑。

非交互安装只装 PC Service：

```bash
KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=service scripts/tool/setup_env.sh
```

如果已经安装过但需要强制重装：

```bash
KITOV_FORCE_XROBOT_SERVICE_INSTALL=1 KITOV_INSTALL_TARGET=skip KITOV_XROBOT_SETUP=service scripts/tool/setup_env.sh
```

启动服务：

```bash
/opt/apps/roboticsservice/runService.sh
```

`runService.sh` 会把 `RoboticsServiceProcess` 放到后台运行，然后返回命令行。

检查服务是否已经运行：

```bash
pgrep -af RoboticsServiceProcess
```

关闭服务：

```bash
pkill -f RoboticsServiceProcess
```

## 运行 XRobot Probe

确认 PC Service 正在运行，并且 PICO 端 app 已连接电脑 IP 后：

```bash
uv run python scripts/debug/xrobot_probe.py --hz 50 --print-every 1
```

健康输出大致应该是：

```text
body_available=True
poses=24
body_hz around 50
Pelvis / Left_Foot / Right_Foot 等 pose 数值持续更新
A/B/X/Y 按钮状态能响应
```

如果 callback 模式不可用，可以强制使用 polling 模式：

```bash
uv run python scripts/debug/xrobot_probe.py --mode polling --hz 50 --print-every 1
```

## 运行在线 GMR 重定向

把 PICO 实时人体数据重定向到 BUMI qpos：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --hz 50 --quiet-gmr
```

把 PICO 实时人体数据重定向到 G1 qpos：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot g1 --hz 50 --quiet-gmr
```

把 PICO 实时人体数据重定向到 OpenArm v1 qpos：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot openarm_v1 --hz 50 --quiet-gmr
```

固定运行 10 秒做测试：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --duration 10 --print-every 1 --quiet-gmr
```

打印完整 qpos：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --print-qpos all --quiet-gmr
```

打开 MuJoCo 可视化：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot bumi --viewer --show-human --quiet-gmr
```

OpenArm v1 可视化：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer --show-human --quiet-gmr
```

只看人体坐标轴，并显示所有 XRobot 关节名：

```bash
uv run python scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer --show-human --show-all-human --human-axes-only --show-human-name --quiet-gmr
```

用保存帧调 OpenArm v1 IK JSON 参数：

```bash
uv run python scripts/debug/tune_xrobot_ik_config.py \
  date/20260725_164701_673558_g1_frame003344.json \
  --robot openarm_v1 \
  --quiet-gmr
```

调参面板里的 `Human Display` 页签可以勾选要显示的人体坐标系；默认只勾选 `ik_match_table1/2` 里参与 IK 的人体关节。

在 viewer 模式里按 `P`，会保存最近一帧 PICO/XRobot 人体数据：

```text
recordings/xrobot_frames/*.json
```

保存的 JSON 包含 SDK 原始 Unity pose、转换后的 GMR pose、时间戳，以及当前重定向出来的 qpos。可以用 `--save-dir` 指定其他保存目录。

把保存下来的一帧重定向到 BUMI 并打开 viewer：

```bash
uv run python scripts/debug/replay_xrobot_frame.py date/20260725_164701_673558_g1_frame003344.json --robot bumi --offset-to-ground --viewer --show-human --quiet-gmr
```

## OpenArm v1 实机接口

OpenArm v1 上实机链路目前只到重定向 qpos，不接 policy：

```text
PICO / XRobot
  -> Kitov_deploy 本地 GMR
  -> openarm_v1 qpos
  -> 安全限幅 / 限速
  -> openarm_can
  -> SocketCAN
  -> Damiao motors
```

硬件映射配置在：

```text
configs/hardware/openarm_v1.json
```

里面定义了每个 MuJoCo 关节对应的 CAN 口、电机类型、发送 ID、接收 ID、方向 `sign`、零点 `zero_offset`、`kp/kd` 和最大速度。`hardware_lower/hardware_upper` 如果是 `null`，会自动使用 XML 里的 joint range，并经过 `sign/zero_offset` 转到硬件坐标；如果手动填写，则以 JSON 里的值为准。`recv_timeout_us` 是普通状态回读等待时间，`enable_recv_timeout_us` 是 enable / disable 后等待电机回包的时间，当前按 OpenArm CLI 的做法设为 500ms。当前默认按实机观察配置为右臂 `can1`、左臂 `can0`，每条 CAN 总线上的电机 ID 是 `0x01..0x07`，接收 ID 是 `0x11..0x17`。这个默认只能作为起点，实机前必须按真实硬件校准。

末端执行器放在独立目录：

```text
ee_body/
  config/openarm_v1_dm_gripper.json
  drivers/openarm_can_gripper.py
```

`configs/hardware/openarm_v1.json` 里只保留一行末端选择。当前选中 OpenArm DM 夹爪：

```json
"ee_body": "openarm_v1_dm_gripper"
```

不接末端时改成：

```json
"ee_body": "none"
```

这样会加载 `ee_body/config/openarm_v1_dm_gripper.json`，并使用
`ee_body/drivers/openarm_can_gripper.py` 里的实现。末端执行器自己的发送位置、读取当前角度、
标零、`kp/kd`、速度和力矩限制都在 `ee_body/` 里维护；手臂的 7DoF 映射仍然只由
`configs/hardware/openarm_v1.json` 负责。
如果末端插件实现了调试面板，`scripts/debug/openarm_hardware_tuner.py` 会自动把它挂到
硬件调参界面里。

OpenArm DM 夹爪配置里，`open_position` 是张开位置，`close_position` 是机械极限闭合
位置，`safe_close_position` 是日常 trigger 控制允许闭合到的最深位置。`close_speed_rad_s`
和 `close_torque_pu` 用于闭合，默认比张开更慢、力矩更小，避免夹碎物体。
`torque_stop_threshold` 如果填数值，会在反馈力矩超过阈值时停止继续闭合；默认 `null`
表示先不用力矩阈值，只靠低力矩限制和卡滞检测。`stall_velocity_threshold` 和
`stall_hold_time_s` 用于检测“目标还在继续闭合，但夹爪速度已经接近 0”的接触状态，检测到后
保持当前夹爪位置，松开 trigger 后解锁。

安装 OpenArm CAN 库前先装系统依赖。你遇到的 `Could not find CLI11`
就是这里缺 `libcli11-dev`：

```bash
sudo apt update
sudo apt install -y cmake build-essential libcli11-dev can-utils
```

如果当前 Ubuntu 源里没有 `libcli11-dev`，先源码安装 CLI11：

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

然后编译并安装 OpenArm CAN 子模块：

```bash
cd third_party/openarm_can
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build

cd python
uv pip install .
```

配置 CAN 口：

```bash
openarm-can-cli -i can0 can_configure
openarm-can-cli -i can1 can_configure
```

推荐的 OpenArm 实机遥操作启动入口会自动检查并配置 `can0/can1`，确认
`XRoboToolkit PC Service` 已启动，然后运行实时控制。`Ctrl+C` 退出时会自动停止
本次控制程序并关闭 `RoboticsServiceProcess`：

```bash
./scripts/run_openarm_teleop.sh
```

默认等价于：

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

额外参数可以直接追加到脚本后面，例如：

```bash
./scripts/run_openarm_teleop.sh --print-targets head
```

可选环境变量：

```bash
KITOV_HZ=50
KITOV_OPENARM_CAN_INTERFACES="can0 can1"
KITOV_OPENARM_CAN_BITRATE=1000000
KITOV_OPENARM_CAN_DBITRATE=5000000
KITOV_XROBOT_SERVICE_SCRIPT=/opt/apps/roboticsservice/runService.sh
KITOV_STOP_ROBOTICS_SERVICE_ON_EXIT=1
```

只读检查电机状态，不会 enable 电机：

```bash
uv run python scripts/debug/openarm_can_probe.py --hz 10 --print-every 1
```

实时 XRobot -> GMR -> OpenArm 硬件目标 dry-run，只打印目标，不 import `openarm_can`，也不发 CAN：

```bash
uv run python scripts/xrobot_openarm_control.py --hz 50 --quiet-gmr --print-targets head
```

带 MuJoCo viewer 的 dry-run。这个 viewer 显示的是经过 `sign / zero_offset /
hardware_lower / hardware_upper / max_velocity_rad_s` 之后的命令 qpos，不是原始
GMR IK qpos：

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

`scripts/debug/xrobot_retarget.py --robot openarm_v1 --viewer` 仍然只用于检查纯 GMR
重定向结果；要看实机会收到的目标，请用上面的 `xrobot_openarm_control.py --viewer`。

真正发送到实机必须显式加 `--send --enable-motors`：

```bash
uv run python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --send \
  --enable-motors
```

如果已经在 `configs/hardware/openarm_v1.json` 里选择了末端执行器，例如
`"ee_body": "openarm_v1_dm_gripper"`，并且要用 PICO trigger 控制夹爪，再额外加
`--enable-ee-control`：

```bash
uv run python scripts/xrobot_openarm_control.py \
  --hz 50 \
  --quiet-gmr \
  --send \
  --enable-motors \
  --enable-ee-control
```

默认左 trigger 控 `left_gripper`，右 trigger 控 `right_gripper`。trigger 松开对应张开，
trigger 按下对应向 `safe_close_position` 闭合；闭合过程会使用夹爪配置里的低速、低力矩和
接触保持逻辑。

脚本会先连接 CAN，然后立刻 `enable_all`。之后会等待一小段时间读取稳定的当前电机
位置，只有启动保持目标通过 `hardware_lower/hardware_upper` 或 XML joint range 检查后
才会发送 MIT hold target。启动读数会先尝试按 `2*pi` 周期折回到有效范围内，例如把
接近编码边界的 `-12.4rad` 折回到当前关节允许范围；如果仍然不在范围内才会拒绝。
默认还会拒绝绝对值超过 `--startup-position-abs-limit 6.283` 的折回后读数。第一帧
XRobot 人体数据到来前，会持续发送当前电机位置作为保持目标；运行中如果突然没有新的人体
数据，也会持续发送上一帧目标，让机械臂保持当前姿势。退出时默认会 `disable_all`。

独立硬件调试界面：

```bash
uv run python scripts/debug/openarm_hardware_tuner.py --hz 50
```

这个界面不接 XRobot/GMR，只调 `configs/hardware/openarm_v1.json -> openarm_can`。
默认 `--slider-space sim`，也就是滑块值就是 MuJoCo 关节角，
发送时再通过 `sign/zero_offset` 转成硬件角。这和实时控制的映射路径一致，适合检查
MuJoCo 命令方向和真实机械臂方向是否一致。每个关节都有当前反馈角、当前反馈转回 MuJoCo
坐标后的角、目标角滑块、最后发送目标、误差和使能状态。`Enable All` 只使能，
`Hold Current` 会把目标滑块同步到当前反馈，`Send Once` 发送一次经过
`max_velocity_rad_s` 限速的目标步进，`Auto Send` 会持续按滑块目标发送。`Flip Sign`
会把对应关节的 `sign` 取反并写回 JSON。`Set Motor Zero All` 调用电机硬件标零，
`Save JSON Zero Offset` 只把当前反馈写进本仓库 JSON 的 `zero_offset`，不改电机内部
零点。

如果 `configs/hardware/openarm_v1.json` 里选择了末端执行器，界面底部会自动出现对应
插件面板。当前 OpenArm DM 夹爪插件会显示 `left_gripper/right_gripper` 的实际角、
速度、力矩、目标滑块，并且可以直接调 `sign`、`safe_close_position`、
`open_speed_rad_s/open_torque_pu`、`close_speed_rad_s/close_torque_pu` 和
`torque_stop_threshold`。`Open` 张开，`Safe Close` 用 `safe_close_position`
低速低力矩闭合，`Close Limit` 才会发机械极限闭合位置。`Flip Sign` 会先改运行时方向，
`Save EE JSON` 才会把这些末端参数写回 `ee_body/config/openarm_v1_dm_gripper.json`。

如果只想做底层电机角测试，可以用硬件角滑块：

```bash
uv run python scripts/debug/openarm_hardware_tuner.py --hz 50 --slider-space hardware
```

## 模型推理

模型文件不提交到本仓库，需要你手动复制导出的文件。

BFM zero / Kitov ONNX bundle：

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

BUMI RGMT ONNX：

```text
models/bumi/rgmt/
  policy.onnx
```

RGMT 的 deploy 配置放在：

```text
configs/policy/bumi_rgmt.json
```

如果目录里同时有 `policy.onnx` 和 `policy.pt`，默认优先使用 `policy.onnx`。
`policy.pt` 仍可作为 fallback，但需要当前 uv 环境里有 PyTorch。

检查 BUMI 模型是否放对：

```bash
uv run python scripts/debug/check_policy_model.py --robot bumi --check-fk
```

运行实时 XRobot -> GMR -> policy 推理，但不打开 MuJoCo 窗口：

```bash
uv run python scripts/xrobot_policy_infer.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --hz 50 \
  --offset-to-ground \
  --quiet-gmr
```

运行实时 XRobot -> GMR -> policy -> MuJoCo sim2sim viewer：

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

这会打开两个窗口：

```text
policy MuJoCo viewer：策略输出 q_target，经 PD torque control 作用到机器人，自动加地板和灯光
GMR viewer：实时 GMR 重定向出来的 reference qpos；带 --show-human 时会叠加人体目标
```

GMR viewer 会放到独立子进程里启动，避免两个 `mujoco.viewer` 在同一个
Python 进程里抢 GLFW/OpenGL 导致段错误。对你来说仍然是一条命令、一个终端启动和关闭。

`--viewer` 模式默认先保持 damping，不让 policy 立刻接管。终端和 policy MuJoCo
viewer 里的按键一致：

```text
p / P: damping 阻尼模式
0:     关节复位到 0，然后保持 damping
1:     policy 接管
```

短启动脚本等价于上面的 BUMI 命令，也会打开两个窗口：

```bash
scripts/run_bumi_policy_sim.sh
```

BUMI RGMT 走另一条模型线，不经过 `backward_encoder.onnx`：

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

短启动脚本：

```bash
scripts/run_bumi_rgmt_policy_sim.sh
```

BUMI RGMT 实机入口使用 Noetix SDK：

```bash
scripts/run_bumi_policy_real.sh
```

这个脚本会先检查 `RoboticsServiceProcess` 是否已经运行；如果没有，会自动执行
`/opt/apps/roboticsservice/runService.sh`。如果服务是本脚本启动的，`Ctrl+C` 退出时会先停止
控制程序，再关闭本次启动的 XRoboToolkit PC Service。需要保留服务不关时：

```bash
KITOV_STOP_ROBOTICS_SERVICE_ON_EXIT=0 scripts/run_bumi_policy_real.sh
```

脚本默认会连接 `third_party/noetix_sdk_bumi`，并发送电机命令。启动后默认是 damping
阻尼状态：

```text
p / P: 回到 damping
0:     按限速把所有关节目标复位到 0
1:     只允许从 0 复位状态进入 RGMT policy；damping 状态下按 1 不会接管
```

实机硬件参数在这里调整：

```text
configs/hardware/bumi_noetix.json
```

其中 `max_velocity_rad_s` 控制 `0` 复位和 policy 目标的每关节限速；
`policy_kp/policy_kd` 是策略接管时发送给 Noetix SDK 的 PD 参数；
`zero_kp/zero_kd` 是按 `0` 复位时使用的 PD 参数。

如果运行中短时间没有新的 XRobot 人体帧，实机入口不会切回 damping；它会继续使用
最后已经进入 RGMT buffer 的 PICO/GMR reference，让 policy 自己保持输出。如果按 `1`
进入 policy 时 RGMT reference buffer 还没攒够，实机会继续保持按 `0` 生成的复位目标；
只有从未进入过复位目标、也无法合法推理时，才会保持 damping。

实时 RGMT 默认会把 policy 当前 reference 延迟 `rgmt_command_window_after` 帧。
当前配置是 10 帧，50Hz 下约 0.2s。这样 command window 的未来 10 帧会使用真实收到的
PICO/GMR reference，而不是复制最新帧。要关掉这个延迟可以显式加：

```bash
scripts/run_bumi_rgmt_policy_sim.sh --reference-delay-frames 0
```

RGMT 输入对齐训练侧四路输入：

```text
rgmt_policy:         projected_gravity + base_ang_vel + dof_pos_rel + dof_vel + last_action
rgmt_state_history:  最近 10 帧 state_obs
rgmt_action_history: 最近 10 帧 action
rgmt_command:        GMR reference 组成的 21 帧窗口，包含 anchor 速度、重力方向和 reference joint pos
```

G1：

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

推理 runtime 按 Kitov 训练导出的 metadata 拼输入：

```text
backward_encoder.onnx:
  state + last_action + privileged_state -> z

policy ONNX:
  actor_obs = metadata 要求的 state / last_action / history_actor + z
  action -> normalized action -> q_target
```

在 `--viewer` 模式里，`q_target` 会通过 PD torque control 作用到同一个
robot XML 上。PD 增益和 torque limit 放在 `configs/policy/g1.json` 和
`configs/policy/bumi.json` 里。这两个文件也保存了训练侧 actuator 的
armature 和 frictionloss，创建 MuJoCo model 后会写回对应 dof。

G1 sim 调试时可以加 `--elastic-band`，这是 UFO-Deploy 里同类的软根部支撑。
默认弹簧参数和 UFO 一致；`--elastic-length 1.5` 是 G1 replay/viewer 测试的
一个可用起点。

## 诊断工具

正常实时推理只打印简洁状态。如果需要看 z、reference 离地高度、policy action
饱和程度等诊断信息，加：

```bash
--debug
```

需要打印 policy 输出的 q_target 时再加：

```bash
--print-q-target head
# 或
--print-q-target all
```

诊断 policy sim 时可以切换 MuJoCo 控制源：

```bash
# 策略输出控制机器人，默认模式
--control-source policy

# 直接用 GMR reference dof_pos 控制机器人，用来排除 policy 输出问题
--control-source reference

# 保持当前关节，用来排除 viewer/sim 本身问题
--control-source hold
```

不用 PICO/GMR 在线流，直接用训练数据里的 BFM motion 做离线 ONNX + PD replay：

```bash
uv run python scripts/debug/replay_bfm_policy.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --data-path /home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/bumi/bumi_lafan_full.pkl \
  --motion-index 0 \
  --max-frames 600
```

G1 不要用 motion index `0` 当稳定性测试，因为它是 fall/get-up 动作。不传
`--motion-index` 或 `--motion-key` 时，脚本会自动跳过明显的 fall/get-up/lie
clip，从第一个普通 motion 开始：

```bash
uv run python scripts/debug/replay_bfm_policy.py \
  --robot g1 \
  --model-dir models/g1/kitov_fb_g1 \
  --data-path /home/unitree/robot_code/Kitov/Kitov/Glush_motion_data/BFM/g1/lafan_29dof.pkl \
  --max-frames 600 \
  --elastic-band \
  --elastic-length 1.5
```

查看可用 motion：

```bash
uv run python scripts/debug/replay_bfm_policy.py --robot g1 --list-motions
```

这个脚本会先把 expert pkl 里的 motion 通过 `backward_encoder.onnx` 预计算成
`z_seq`，再逐帧执行 `policy ONNX -> q_target`。

打开 policy sim 和 reference motion 两个 viewer：

```bash
uv run python scripts/debug/replay_bfm_policy.py \
  --robot bumi \
  --model-dir models/bumi/kitov_fb_bumi_action_scale_0.5 \
  --motion-index 0 \
  --viewer \
  --reference-viewer
```

只有想加载原始 robot XML、不加地板和灯光时，才使用 `--no-floor`。按终端里的
`Ctrl-C` 会关闭两个 viewer 并停止 XRobot stream。
