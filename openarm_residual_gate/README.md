# Residual zero-gate — ReBot B601-DM (default) + OpenArm archive

对齐群契约、`docs/ReBot_B601_DM_Survey.md`、`docs/OpenArm_Residual_RL_Training_Guide.md`。

## 默认路径：ReBot B601-DM

| 项 | 值 |
|---|---|
| 契约 | **`contract_rebot_b601_dm.json`**（独立文件；**不**覆盖 OpenArm `contract.json`） |
| 机器人 | `rebot_b601_dm`，`n_dof=7`（J1–6 + `finger_left`） |
| `obs_dim` | 7+7+7+1=**22**；Actor `obs_keys=[q,qdot,a_base,phase]` |
| `control_hz` | **20**（residual 控制步；≠ 真机/ROS 常写的 100 Hz） |
| `sim_dt` / `n_substeps` | 0.001 / 50 |
| `gripper_unit` | **`m`**（MJCF `finger_left` 0…0.05）；`lerobot_gripper_unit=deg` 仅文档占位，接 MEAN_STD 前禁止静默混单位 |
| `action_dim_map` | 契约内 7 维表（名+单位+range） |
| `stats_id` | `pending_rebot_b601_dm`，`stats_source=none` |
| 场景 | `sim/rebot_b601/scene_residual.xml`（本地 overlay；vendor STL `nu=0` 未改） |
| 方块 | `red_cube`（官方 STL 已有） |
| 相机 | `overhead`←官方 `overhead_rgb`；`wrist` 键冻结为 `observation.images.wrist` 但 **MJCF 外参未发明** |
| `obs_keys_hash` | `b8b5eef23fc3dd73`（含相机 digest；wrist 外参 null/pending） |
| vendor | `vendor/rebot_arm_digitaltwin_dm` @ `7456d8b` |

### 运行 ReBot 门禁

```bash
cd /workspace/openarm_smolvla_residual
source .venv/bin/activate
export MUJOCO_GL=egl
python3 openarm_residual_gate/test_zero_residual_gate_mujoco.py
# 日志：logs/rebot_gate/{gate_stdout.txt,gate_meta.json,obs_keys_hash.txt}
```

实现：`residual_env_rebot_mujoco.py`、`fake_base.py(n=7)`、`obs_keys_hash.py`。

范式：`a_exec=clip(a_base+δ)`，`δ=0≡base`，FakeBase，图像不进 Actor。

## Path A draft: LIBERO Spatial (pure sim)

| 项 | 值 |
|---|---|
| 契约 | **`contract_libero_spatial_v0.json`**（独立；**不**覆盖 ReBot / OpenArm） |
| 后端 | **stub**（draft pending full libero install）；真环境见 `scripts/attempt_libero_real_env.py` |
| `action_dim` | **7** relative EE + gripper（`normalized_relative_ee`；**非** ReBot joint rad） |
| `proprio_dim` | **8** = eef_pos(3)+axis_angle(3)+gripper(2) → LeRobot `observation.state` |
| `obs_dim` | **16** = proprio(8)+a_base(7)+phase(1)；图像**不**进 Actor |
| `clip_c` | **0.05**（`clip_c_unit=normalized_relative_ee`；独立于 ReBot ±0.03 rad） |
| `control_hz` | **20**（robosuite/LIBERO native；残差只在 control step） |
| `reward_version` | `libero_v0_sparse_success_plus_delta_l2` |
| success | **`env.check_success()` only** — 禁止 z_lift |
| `failure_mode` | {none, timeout, unsafe} |
| `stats_id` | `pending_libero`（正式 MEAN_STD 待从 lerobot/libero 或 HuggingFaceVLA/libero 拉取） |
| VLA keys | agentview→`agentview_image`→`observation.images.image`；wrist→`robot0_eye_in_hand_image`→`observation.images.image2` |
| `obs_keys_hash` | `d9807cfaf4a8701c` |
| task | `libero_spatial` task_id=0 |
| 门禁 | `test_zero_residual_gate_libero.py` → **PASS (STUB)**；日志 `logs/libero_gate/` |

### 运行 LIBERO 门禁（stub）

```bash
cd /workspace/openarm_smolvla_residual
source .venv/bin/activate
python3 openarm_residual_gate/test_zero_residual_gate_libero.py
# 日志：logs/libero_gate/{gate_stdout.txt,gate_meta.json,obs_keys_hash.txt}
```

实现：`residual_env_libero.py`、`fake_base.py`（`LIBERO_NOOP_ACTION`）、`obs_keys_hash.py`（可选 `proprio_dim`）。
ReBot `contract_rebot_b601_dm.json` / hash `b8b5eef23fc3dd73` **未改**。

## 归档：OpenArm v1

| 项 | 值 |
|---|---|
| 契约 | `contract.json`（仍为 `openarm_v1_residual_env_v0`，**未覆盖**） |
| 门禁 | stub：`test_zero_residual_gate.py`；MuJoCo 归档：`test_zero_residual_gate_openarm_mujoco.py` |
| Env | `residual_env_mujoco.py` / `residual_env_stub.py` |
| `obs_keys_hash` | `881c67cd24c3a672`（stats_id=`pending_single_arm`） |
| 相机 | head + wrist（OpenArm cell 外参）；≠ ReBot overhead |

## 已知限制

- ReBot wrist MJCF 外参未锁定；门禁可不渲 wrist
- FakeBase 未接冻结 SmolVLA；reward 仍为 `-‖δ‖²` placeholder
- `nq=29`（含 freejoint 物块）；Actor 只用 7 DoF
- 未开 PR / 未长训
