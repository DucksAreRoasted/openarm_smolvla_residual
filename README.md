# openarm_smolvla_residual

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
