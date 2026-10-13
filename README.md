# openarm_smolvla_residual

## 当前进展（2026-10-13）

主路径是 **ManiSkill PickCubeReBot-v1 + ReBot B601-DM + SmolVLA**。
已完成 100 条 yaw 对齐示范转换（18,048 帧）及 10,000 步正式微调
（batch=8，seed=1000，state/action 均为 7 维，最终 loss=0.015）。
短预算闭环测试在 seed 10000–10004 上成功 5/5，Wilson 95% 区间为
[0.5655, 1.0000]；正式 B0 判定和 ManiSkill 残差 PPO 尚未完成。
夹爪原始预测仍有小幅边界偏差，详见[正式模型测试记录](docs/正式模型测试记录_s10000.md)。

代码、配置、执行手册和小型结果摘要纳入 Git；虚拟环境、数据集、模型权重、
原始日志、第三方仓库、STL 网格和传输压缩包保留在本地。
可分享的训练及评测摘要见 [reports/s10000](reports/s10000)。

### 克隆后恢复仿真资产

```bash
# 在已有 trimesh 的仿真环境中运行；锁定版本以复现规划网格
uv pip install --python .venv-ms3/bin/python trimesh==5.1.1
.venv-ms3/bin/python scripts/prepare_rebot_assets.py
```

该命令下载固定提交的上游 ReBot 仓库并生成规划网格，最后校验
`configs/rebot_asset_manifest.json` 中的 SHA256。已有资产可用
`python3 scripts/prepare_rebot_assets.py --check-only` 校验。
这些资产不会提交到本仓库。归档 OpenArm 路径的资产来源见 `vendor/VERSIONS.txt`。

环境、数据转换与评测步骤依次见：

- [LeRobot 数据转换与微调](docs/LeRobot数据转换与SmolVLA微调执行手册.md)
- [统一仿真环境](docs/统一仿真环境与冒烟测试执行手册.md)
- [B0 基线评测](docs/B0基线评测执行手册.md)
- [残差 PPO 冒烟](docs/残差PPO冒烟训练执行手册.md)

正式训练参数保存于 `configs/smolvla_rebot_train_s10000.json`。
恢复数据并完成手册中的环境准备后，可运行
`.venv-lerobot/bin/python scripts/run_smolvla_formal_rebot.py --detach`；
已有输出或运行记录时会拒绝重复启动。模型权重和数据集需自行恢复或重新生成。

## 早期门禁与归档路径（2026-10-12）

**Default residual gate (2026-10-12): ReBot B601-DM** (`n_dof=7`), not OpenArm.
See `docs/ReBot_B601_DM_Survey.md`, `openarm_residual_gate/contract_rebot_b601_dm.json`,
`sim/rebot_b601/scene_residual.xml`. Gate log: `logs/rebot_gate/` — `obs_keys_hash=b8b5eef23fc3dd73` **PASS**.
OpenArm `contract.json` / hash `881c67…` kept as archive.

---

OpenArm v1 单臂 + 冻结 SmolVLA + Residual PPO（抓取抬升）。

## 当前状态（2026-10-12）

- **Path A (算法主路径) LIBERO draft**：`openarm_residual_gate/contract_libero_spatial_v0.json`
  - Actor obs `[proprio(8), a_base(7), phase(1)]` → `obs_dim=16`；`action_dim=7` relative EE
  - success=`env.check_success()`；`reward_version=libero_v0_sparse_success_plus_delta_l2`
  - VLA keys 锁定：agentview→`observation.images.image`，wrist→`observation.images.image2`，proprio→`observation.state`
  - 门禁 stub：**PASS** `obs_keys_hash=d9807cfaf4a8701c`；日志 `logs/libero_gate/`
  - 真 LIBERO 未装（`libero` pip 过重）；`scripts/attempt_libero_real_env.py` 记录 install gap
  - ReBot 契约/hash `b8b5eef23fc3dd73` **冻结未改**
- 训练指导：`docs/OpenArm_Residual_RL_Training_Guide.md`
- 零残差门禁：`openarm_residual_gate/`
  - stub：`test_zero_residual_gate.py` → **PASS**
  - MuJoCo：`test_zero_residual_gate_mujoco.py` → **PASS**（真 `mj_step`）
- 契约：`openarm_residual_gate/contract.json`
  - `camera_in_env=true`：head + wrist，原生 **640×480** RGB，目标 **30 fps**
  - LeRobot keys：`observation.images.head` / `observation.images.wrist`（去掉 top/ceiling）
  - `camera_in_actor_obs=false`：Actor 仍只吃 `[q,qdot,a_base,phase]`（不静默扩维）
  - 外参：`head←v2/cell camera_head_right`，`wrist←camera_wrist_right`
  - `obs_keys_hash`（含相机布局 digest）：`881c67cd24c3a672`（contract sha16=`c0a0faeba463612c`）
- 资产：`vendor/openarm_mujoco` @ `0e050bb`（`0e050bb5d3472b75c6bfa00b310a77c7c2c5be79`）+ 本地相机 overlay
- 依赖：`requirements-gate.txt`（mujoco==3.15.0）；venv：`.venv`

## LeRobot / SmolVLA 环境

训练和闭环评测统一使用 `.venv-lerobot`（Python 3.12），依赖见
`requirements-lerobot.txt`；运动规划采集使用 `.venv-ms3`，避免 mplib 的 NumPy 版本冲突。
包含 SmolVLA、数据集转换和训练依赖。安装方式遵循
[LeRobot 官方文档](https://huggingface.co/docs/lerobot/installation)。

```bash
# 已安装环境：
source .venv-lerobot/bin/activate
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobot OK')"

# 重建环境（需 uv）：
uv venv --python 3.12 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python -r requirements-lerobot.txt
```

视频数据集还需系统 FFmpeg（Ubuntu：`sudo apt-get install ffmpeg`）。
若终端继承了 ROS 的 `PYTHONPATH` 并出现包冲突，可用
`env -u PYTHONPATH .venv-lerobot/bin/python ...` 运行。
安装不包含模型权重；GPU 使用还需可用的 NVIDIA 驱动。
已在 RTX 5070 Laptop 上完成 cu128 / sm_120 GPU 仿真、训练及闭环评测。
ManiSkill 必须按统一环境手册使用 `--no-deps` 安装，再补齐除 mplib 外的依赖。

## 复现门禁

```bash
cd /workspace/openarm_smolvla_residual
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gate.txt
# 无头 GL（Debian/Ubuntu）：sudo apt-get install -y libegl1 libegl-mesa0
export MUJOCO_GL=egl

python3 openarm_residual_gate/test_zero_residual_gate.py
python3 openarm_residual_gate/test_zero_residual_gate_mujoco.py
python3 openarm_residual_gate/test_zero_residual_gate_libero.py  # Path A LIBERO stub
```

## 相机渲染 smoke（非完整 VLA 观测管线）

只验证 EGL 能渲染 head/wrist 各一帧且非全黑；**不会**把图像塞进 Actor obs。

```bash
cd /workspace/openarm_smolvla_residual
source .venv/bin/activate
export MUJOCO_GL=egl
python3 scripts/render_cameras_smoke.py
# 输出：logs/camera_smoke/{head,wrist}.{png,npy} 与 smoke_meta.json
```

## 已知缺口

- 无 cube 抓取场景 / 稀疏 success 奖励
- 图像未接入 Env `info` / LeRobot 数据集写出 / 冻结 SmolVLA
- FakeBase → 冻结 SmolVLA 未接
- 门禁六条中特权隔离 / B0 落盘 / 完整 reward v0 仍待后续
- 未验证持续 30 fps 吞吐（smoke 仅单帧）

## 下一步

1. 加桌面+方块场景与稀疏 success 奖励
2. 锁定单臂 MEAN_STD id，接冻结 SmolVLA chunk（图像走 LeRobot keys）
3. 真 Env 上补齐门禁六条；通过后再极短 PPO smoke

详见训练指导文档与群聊契约。**不要**长时间 PPO；本阶段不开 PR。
