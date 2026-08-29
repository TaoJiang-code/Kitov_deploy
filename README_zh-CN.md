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

在线重定向目前支持：

```text
g1 / unitree_g1
bumi
openarm / openarm_v1
```

## 初始化 Submodule

G1、BUMI 和 OpenArm v1 的机器人 XML / mesh 来自本仓库的 `Glush_Zoo` submodule。
OpenArm v1 的 CAN SDK 来自 `third_party/openarm_can` submodule：

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

## 创建 uv 环境

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

新设备上先在仓库主目录的 `workspace/xrobot_toolkit/` 里准备两个仓库。这个目录已加入
`.gitignore`，不会提交到仓库。下面每个代码块都默认从 Kitov_deploy 仓库主目录执行：

```bash
mkdir -p workspace/xrobot_toolkit
cd workspace/xrobot_toolkit

git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
```

编译 XRoboToolkit C++ SDK：

```bash
cd workspace/xrobot_toolkit/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
```

把 C++ SDK 产物放进 Python binding 项目：

```bash
cd workspace/xrobot_toolkit/XRoboToolkit-PC-Service-Pybind
mkdir -p lib include

cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
```

安装到 Kitov_deploy 的 uv 环境：

```bash
uv pip install workspace/xrobot_toolkit/XRoboToolkit-PC-Service-Pybind
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

根据系统版本安装对应包。本机是 `Ubuntu 20.04.6`，使用：

```bash
sudo dpkg -i packages/xrobotoolkit_pc_service/XRoboToolkit_PC_Service_1.0.0_ubuntu_20.04_amd64.deb
```

如果是在 Ubuntu 22.04 机器上，安装 22.04 包：

```bash
sudo dpkg -i packages/xrobotoolkit_pc_service/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

如果 `dpkg` 提示系统依赖缺失：

```bash
sudo apt-get install -f
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

里面定义了每个 MuJoCo 关节对应的 CAN 口、电机类型、发送 ID、接收 ID、方向 `sign`、零点 `zero_offset`、`kp/kd` 和最大速度。`hardware_lower/hardware_upper` 如果是 `null`，会自动使用 XML 里的 joint range，并经过 `sign/zero_offset` 转到硬件坐标；如果手动填写，则以 JSON 里的值为准。`recv_timeout_us` 是普通状态回读等待时间，`enable_recv_timeout_us` 是 enable / disable 后等待电机回包的时间，当前按 OpenArm CLI 的做法设为 500ms。当前默认是假设右臂在 `can0`、左臂在 `can1`，每条 CAN 总线上的电机 ID 是 `0x01..0x07`，接收 ID 是 `0x11..0x17`。这个默认只能作为起点，实机前必须按真实硬件校准。

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
默认会打开一个由实机反馈驱动的 MuJoCo viewer，用来检查真实机械臂和 MuJoCo 里的关节
转动方向是否一致。每个关节都有当前反馈角、当前反馈转回 MuJoCo 坐标后的角、目标角滑块、
最后发送目标、误差和使能状态。`Enable All` 只使能，`Hold Current` 会把目标滑块同步
到当前反馈，`Send Once` 发送一次经过 `max_velocity_rad_s` 限速的目标步进，`Auto Send`
会持续按滑块目标发送。`Flip Sign` 会把对应关节的 `sign` 取反并写回 JSON，只改变
反馈到 MuJoCo 的方向映射，不改变当前界面滑块的硬件范围。`Set Motor Zero All` 调用电机
硬件标零，`Save JSON Zero Offset` 只把当前反馈写进本仓库 JSON 的 `zero_offset`，不改
电机内部零点。

## 模型推理

模型文件不提交到本仓库，需要你手动复制导出的 ONNX bundle：

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

`--viewer` 模式默认先保持 damping，不让 policy 立刻接管。把焦点放在终端按
`p`，或者在 policy MuJoCo viewer 里按 `P`，可以在 damping 和 policy 控制之间切换。

短启动脚本等价于上面的 BUMI 命令，也会打开两个窗口：

```bash
scripts/run_bumi_policy_sim.sh
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
