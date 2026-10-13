# ReBot B601-DM 调研报告 — 替换 OpenArm v1 单臂（Residual PPO + 冻结 SmolVLA）

**日期：** 2026-10-12（Asia/Shanghai）  
**范围：** 文档 + 轻量验证（MJCF 可加载性）。**未改写** `openarm_residual_gate` / OpenArm 工作代码；**未删** `vendor/openarm_mujoco`。  
**目标工程：** `/workspace/openarm_smolvla_residual`  
**本地 clone（可选）：** `vendor/rebot_arm_digitaltwin_dm` @ `7456d8bc3cc6d9c470680988543f90bc8f3b03b0`

---

## 0. 一句话结论

**B601-DM = 达妙（Damiao）电机版 6-DoF + 夹爪桌面臂；官方 MuJoCo / URDF / ROS2 / LeRobot / Isaac 齐全，但上游 MJCF `nu=0`（无 position actuator），且仿真相机只有 `overhead_rgb`（≠ OpenArm `head`），手腕 D405 仅硬件/LeRobot，无官方 MJCF wrist cam。** 切到 ReBot 后 residual 门禁契约形状可保留，但 **`n_dof` 8→7**、关节名、FakeBase home、MJCF + 本地 actuator overlay、相机键名必须一起改。

---

## 1. B601 vs B601-DM vs B601-RS（「DM」含义）

| 型号 | 「后缀」含义 | 电机 | 供电 | 负载 | 外观 |
|------|-------------|------|------|------|------|
| **B601-DM** | **DM = Damiao（达妙）** | J1–3: DM4340P；J4–6+夹爪: DM4310 | **DC 24V** | **1.5 kg** | 与 RS 同外观 |
| **B601-RS** | **RS = RobStride（灵足）** | RobStride 系列 | **DC 48V** | **2.5 kg** | 同外观 |
| 「B601」 | 产品族名；购买/文档几乎都带 DM 或 RS | — | — | — | — |

**逐字事实（官方 README）：**

> Two arm models — All design files for the **B601-DM (Damiao)** and **B601-RS (Robstride)**, two arms sharing the same appearance.

来源：https://github.com/Seeed-Projects/reBot-DevArm

**电机配置（DigitalTwin_DM README 原文）：**

> Motors: Damiao 4340P (joint1-3) + 4310 (joint4-6 + gripper)  
> Communication: USB-CAN (`/dev/ttyACM0`)

来源：https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM

**关节极限（`ReBot_Arm_DM.urdf`，本地 clone）：**

| Joint | type | limit |
|-------|------|-------|
| joint1 | revolute | −2.8 … 2.8 |
| joint2 | revolute | −3.14 … 0 |
| joint3 | revolute | −3.14 … 0 |
| joint4 | revolute | −1.87 … 1.57 |
| joint5 | revolute | −1.57 … 1.57 |
| joint6 | revolute | −3.14 … 3.14 |
| finger_left | prismatic | **0 … 0.05 m** |
| finger_right | prismatic | −0.05 … 0（`mimic` finger_left × −1） |

Web 控制台把夹爪显示成 **0–0.09 m** 开合宽度；电机固件侧还有 rad 映射（ROS `HardwareManager` 做换算）。**Residual 仿真建议以 MJCF/URDF 的 `finger_left` 米制为准（0–0.05）。**

---

## 2. 官方 / 开源资源一览（URLs）

### 2.1 硬件与总仓

| 资源 | URL | 许可 / 备注 |
|------|-----|-------------|
| **reBot-DevArm（硬件+BOM+相机支架）** | https://github.com/Seeed-Projects/reBot-DevArm | 硬件 **CERN-OHL-W-2.0**；软件说明 **Apache-2.0** |
| B601-DM BOM | https://github.com/Seeed-Projects/reBot-DevArm/blob/main/hardware/reBot_B601_DM/readme.md | Damiao 43 系列 |
| 相机支架（DM/RS 共用） | https://github.com/Seeed-Projects/reBot-DevArm/tree/main/hardware/camera-mounts | D405 / D435i / Gemini / UVC |
| 产品页 B601-DM | https://www.seeedstudio.com/reBot-Arm-B601-DM-Assembled-Kit-with-Power-Supply-Bundle.html | 套件 |

### 2.2 仿真 / 数字孪生（**本调研主资产**）

| 资源 | URL | 内容 |
|------|-----|------|
| **ReBot_Arm_DigitalTwin_DM** | https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM | ROS2 + MuJoCo MJCF + URDF + Fake Driver + Web（**推荐 clone**） |
| 镜像/旧名 Borot-Arm_Mujoco | https://github.com/Yang-Ci/Borot-Arm_Mujoco | 同系 DM 数字孪生说明 |
| ReBot_Arm_DigitalTwin_**RS** | https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS | RS 版（用户给的 wiki 链到 RS；**硬件选型是 DM**） |
| Seeed Wiki DM Web/Sim | https://wiki.seeedstudio.com/rebot_arm_b601_dm_web_simulator_developer_guide/ | 安装 / Fake / sim / web |
| Seeed Wiki **RS** Web/Sim（用户指定 URL） | https://wiki.seeedstudio.com/rebot_arm_b601_rs_web_simulator_developer_guide/ | **RS** 栈；结构类似但电机/命名空间不同 |

### 2.3 ROS2 / SDK / Isaac / LeRobot

| 资源 | URL |
|------|-----|
| Seeed 官方 ROS2 | https://github.com/Seeed-Projects/reBotArmController_ROS2 |
| Wiki ROS2 集成 | https://wiki.seeedstudio.com/rebot_arm_b601_dm_ros2_integration/ |
| Python SDK `reBotArm_control_py` | https://github.com/Seeed-Projects/reBotArm_control_py |
| Wiki Pinocchio | https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/ |
| Isaac Sim 工程 | https://github.com/hopcan/reBot-Isaacsim · Wiki https://wiki.seeedstudio.com/rebot_arm_b601_dm_isaacsim/ |
| LeRobot（Seeed fork） | https://github.com/Seeed-Projects/lerobot · Wiki https://wiki.seeedstudio.com/rebot_arm_b601_dm_lerobot/ |
| PyPI：`lerobot-robot-seeed-b601`、`motorbridge` | pip；`--robot.type=seeed_b601_dm_follower`、`--robot.can_adapter=damiao` |
| 视觉抓取 | https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/ |
| MotorBridge | https://motorbridge.seeedstudio.com |

### 2.4 用户指定 URL 核对

1. **`rebot_arm_b601_rs_web_simulator_developer_guide`** → **B601-RS** 数字孪生（RobStride + SocketCAN `can0` 1Mbps + `/rebotarm_rs` 仿真命名空间）。**不是 DM。** 对标 DM 请用 §2.2 DM wiki。  
2. **`Seeed-Projects/reBot-DevArm`** → 硬件/BOM/相机/路线图总仓（**不是** MuJoCo MJCF 主仓）。MJCF 在 **Yang-Ci/ReBot_Arm_DigitalTwin_DM**。

---

## 3. DoF / 动作空间 / 与 OpenArm v1 对比

| 项 | OpenArm v1（当前门禁） | ReBot B601-DM |
|----|------------------------|---------------|
| 臂关节 | **7**（`openarm_joint1`…`7`） | **6**（`joint1`…`joint6`） |
| 夹爪 | `openarm_finger_joint1`（+ equality `finger_joint2`） | `finger_left`（+ mimic/`finger_right`） |
| **契约 `n_dof` / `action_dim`** | **8** | 建议 **7**（J1–6 + finger_left） |
| `obs_dim`（q+qdot+a_base+phase） | 8+8+8+1=**25** | 7+7+7+1=**22** |
| 控制频率参考 | residual `control_hz=20` | 真机 bringup 常 100 Hz；ROS joint_state 默认 100；sim Fake ~100；MIT RS 文档 125（DM 常用 POS_VEL） |
| `sim_dt` | 0.001 | 官方 STL/colored MJCF **`timestep="0.001"`**；simple/kinematic 为 0.002 |
| 执行器 | MJCF 有 `joint*_ctrl` / `finger_ctrl`（`nu=8`） | **上游 MJCF `nu=0`**（用 `qfrc_applied` PD 或写 qpos） |

**LeRobot 动作：** follower `seeed_b601_dm_follower`；遥操作 leader `rebot_arm_102_leader`；相机键自由配置（`front`/`side`/`d405_*`/`orbbec_*`），**没有强制 `head`/`wrist` 键名**。

---

## 4. MuJoCo / URDF 资产（本地验证）

### 4.1 Clone 路径

```text
vendor/rebot_arm_digitaltwin_dm/                    # git clone --depth 1
  reBotArm_ros2_DM/src/rebotarm_mujoco/models/
    simple_rebotarm.xml          # 胶囊几何，无 mesh，可直接 load
    rebotarm_b601_kinematic.xml  # 简化盒几何 + finger_left/right
    rebotarm_b601_stl.xml        # STL 视觉；meshdir=../meshes
    rebotarm_b601_colored.xml    # 默认「真机配色」；meshdir=../meshes
  reBotArm_ros2_DM/src/rebotarm_bringup/description/
    urdf/ReBot_Arm_DM.urdf
    meshes/                      # STL/DAE
```

**本机已做轻量修复（非破坏上游逻辑）：** 将  
`.../rebotarm_mujoco/meshes` → 符号链接到 `.../rebotarm_bringup/description/meshes`，以便 `meshdir="../meshes"` 能加载。  
Commit pin：`vendor/rebot_arm_digitaltwin_dm_COMMIT.txt` = `7456d8b…`。

### 4.2 加载冒烟（2026-10-12，项目 `.venv` + mujoco 3.15.0）

| 文件 | 结果 | nq | nu | ncam | 关节名 |
|------|------|----|----|-------|--------|
| `simple_rebotarm.xml` | OK | 28* | **0** | 1 | joint1–6, finger_left |
| `rebotarm_b601_kinematic.xml` | OK | 29* | **0** | 1 | + finger_right |
| `rebotarm_b601_stl.xml` | OK（需 meshes 链接） | 29* | **0** | 1 | + finger_right |
| `rebotarm_b601_colored.xml` | OK（需 meshes 链接） | 29* | **0** | 1 | + finger_right |

\*含桌面物体 `freejoint` 与任务体。  
**无 `<actuator>`、无 `home` keyframe。** 官方 `mujoco_torque_control.py` 用 `qfrc_applied` + PD；`real2sim_sync` 直接写 `qpos`。

### 4.3 官方 overhead 相机（逐字）

来自 `rebotarm_b601_colored.xml` / `stl` / `simple`：

```xml
<camera name="overhead_rgb" mode="fixed" pos="0.42 0 0.86" xyaxes="1 0 0 0 1 0" fovy="50"/>
```

ROS topic（DM sim）：`/rebotarm/mujoco/overhead_rgb/image_raw`  
（RS 文档用 `/rebotarm_rs/mujoco/overhead_rgb/image_raw`。）

**与 OpenArm 对照（见 `docs/Camera_Layout_Survey.md`）：**

- ReBot **`overhead_rgb` ≠ OpenArm `head`**（OpenArm head 是 cell 夹具立体相机，≈ chest/fixture）。  
- ReBot **官方 MJCF 无 wrist 相机**；硬件有 **D405 腕部支架**（DM/RS 共用 STEP）。  
- LeRobot 示例键：`d405_color` / `d405_depth`、`front` / `side`、Orbbec 等——**不是** `observation.images.head`。

---

## 5. 相机布局建议（切 ReBot 后）

| 角色 | 推荐键名（契约） | 来源 | 备注 |
|------|------------------|------|------|
| 外部/桌面全局 | `overhead`（或保留别名 `head` **但必须文档声明 = overhead≠OpenArm head**） | MJCF `overhead_rgb` | 已有官方外参 |
| 腕部 | `wrist` | **暂无官方 MJCF**；硬件 D405；需本地 overlay 估外参或先 `camera_in_env` 只开 overhead | 与现 OpenArm wrist overlay 同类风险 |

**Actor：** 继续 `camera_in_actor_obs=false`，obs 仍为 `[q,qdot,a_base,phase]`（维数随 `n_dof` 变）。

---

## 6. ROS2 / FakeBase / LeRobot 钩子（集成相关）

### 6.1 ROS2（DM）

- 默认命名空间：`/rebotarm`  
- Fake：`ros2 launch rebotarm_bringup fake_bringup.launch.py`  
- 真机：`./rebotarm start dm`（DigitalTwin）或 `bringup.launch.py channel:=/dev/ttyACM0`  
- 关键接口：`/joint_states`、`/enable`、`FollowJointTrajectory`、`gripper/set`、重力补偿服务  

### 6.2 与本仓库 `FakeBase` 的对应

当前 `fake_base.py`：`n=8`，home ≈ `[0,0,0, π/2, 0,0,0, 0.044]`（OpenArm）。  

切 ReBot 后建议：

```text
n = 7
_home ≈ [0, 0, 0, 0, 0, 0, 0.0]   # 或轻微展开夹爪 e.g. 0.04 m
joint_names = [joint1..joint6, finger_left]
```

chunk 形状 `(T, 7)`；**零残差门禁**：`δ=0 ⇒ a_exec == a_base` 逻辑不变，只改维数与关节映射。

### 6.3 LeRobot

```bash
# 校准 / 遥操（wiki 原文要点）
lerobot-calibrate --robot.type=seeed_b601_dm_follower --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 --robot.can_adapter=damiao
# 训练支持 ACT / SmolVLA / Pi0 …（Seeed fork + wiki）
```

仓库：https://github.com/Seeed-Projects/lerobot · 包：`lerobot-robot-seeed-b601`

---

## 7. 许可（逐字摘要）

| 层 | 许可 |
|----|------|
| 硬件设计（reBot-DevArm） | **CERN-OHL-W-2.0**（弱互惠；商业可用，分发需 Complete Source 等） |
| 软件 SDK（官方声明） | **Apache-2.0** |
| DigitalTwin_DM LICENSE 文件 | **CERN-OHL-W-2.0**（仓库根 LICENSE） |

使用 MJCF/URDF 进本仓库时保留版权与许可声明；不要静默剥离 LICENSE。

---

## 8. 已知缺口（Gaps）

1. **上游 MJCF 无 actuator（`nu=0`）** → 现有 `residual_env_mujoco.py` 的 `data.ctrl ← a_exec` 路径**不能直接**接官方 XML。  
2. **无 `home` keyframe** → `_resolve_home_key()` 需改（设 keyframe 或 reset 到零位 qpos）。  
3. **无 MJCF wrist 相机** → 与「head+wrist」研究叙事冲突；overhead≠head。  
4. **DoF 8→7** → `contract.json`、`obs_keys_hash`、FakeBase、测试断言、未来 MEAN_STD 全部作废/重算。  
5. **夹爪单位混用**：MJCF 0–0.05 m；web 0–0.09 m 开合；电机 rad。门禁必须钉死一种。  
6. **场景含 freejoint 物块**（红/蓝/黄）→ `nq` 远大于 7；Actor 只读臂关节，但 reset/keyframe 要小心。  
7. **RS wiki ≠ DM 硬件**；误跟 RS 会拿到 RobStride/SocketCAN/`/rebotarm_rs`。  
8. **Isaac USD** 在第三方 `hopcan/reBot-Isaacsim`，本 residual 路径优先 MuJoCo，暂不依赖 Isaac。  
9. OpenArm 零残差已 PASS；切臂后需 **新契约版本号**，勿覆盖旧 contract 语义。

---

## 9. 集成方案（保持 residual 门禁契约「形状」）

契约形状 =：`a_exec = clip(a_base + δ)`、Actor obs=`[q,qdot,a_base,phase]`、`camera_in_actor_obs=false`、零残差 `δ=0` 冒烟。**换机器人时改内容，不改这条公式。**

| 模块 | 改什么 | 不动什么 |
|------|--------|----------|
| **MJCF** | 新 scene：include/colored 或 kinematic + **本地 overlay 加 `<position>` actuators ×7** + 可选 `equality`/`tendon` 耦合 `finger_right`；`meshdir` 指到 vendor meshes；可选 strip freejoint 物块做最小门禁场景 | 不删 `vendor/openarm_mujoco` |
| **ActionAdapter / contract** | `n_dof=7`；`joint_names`/`actuator_names`；`obs_dim=22`；`finger_bound_m`≤0.05；新 `stats_id`；重算 `obs_keys_hash` | residual clip 调度思想、chunk_size、control_hz 可先沿用 20 Hz |
| **camera keys** | `cameras[]`：至少 `overhead_rgb`→契约名建议 `overhead`（或文档化 alias `head`）；wrist 第二阶段再加 | Actor 仍不吃图 |
| **FakeBase** | `n=7`，home 全零或 ReBot 安全姿态；chunk `(T,7)` | `infer_chunk` 接口 |
| **Env 类** | 新 `ReBotResidualEnvMujoco`（或参数化 `robot=rebot_b601_dm`）；去掉 OpenArm `finger_joint2` 特例，改为 ReBot finger_right 耦合 | 零残差测试脚本结构 |
| **Vendor** | 保留 OpenArm；ReBot clone 已在 `vendor/rebot_arm_digitaltwin_dm` | — |

---

## 10. 推荐 M0（最小变更拿到 ReBot 零残差 smoke）

**目标：** `δ=0` 时 `‖a_exec − a_base‖∞ < atol`，真 `mj_step`，**不训练、不改 OpenArm 路径。**

### 步骤（建议顺序）

1. **本地 overlay MJCF**（新建，例如 `sim/rebot_b601_dm/scene_gate.xml`）：  
   - include 或复制 `rebotarm_b601_kinematic.xml`（无 STL 依赖，最快）或 colored（更真）；  
   - 追加：
     ```xml
     <actuator>
       <position name="joint1_ctrl" joint="joint1" kp="200" ctrlrange="-2.8 2.8"/>
       <!-- joint2..joint6 同理，ctrlrange=URDF limits -->
       <position name="finger_left_ctrl" joint="finger_left" kp="50" ctrlrange="0 0.05"/>
     </actuator>
     <equality> <!-- 或 tendon：finger_right = -finger_left -->
       <joint joint1="finger_left" joint2="finger_right" polycoef="0 -1"/>
     </equality>
     <keyframe>
       <key name="home" qpos="…全零臂 + 物体初始…"/>
     </keyframe>
     ```
   - 门禁场景可暂时去掉三个 freejoint 物块以降低 `nq` 复杂度（可选）。
2. **新 `contract_rebot_b601_dm.json`**（勿覆盖 OpenArm contract）：`n_dof=7`，相机先只登记 `overhead`（`camera_in_env` 可 true，但不进 Actor）。  
3. **复制/参数化** `residual_env_mujoco.py` + `fake_base.py(n=7)` + `test_zero_residual_gate_mujoco.py` 指向新 contract/XML。  
4. 跑：  
   ```bash
   cd /workspace/openarm_smolvla_residual && source .venv/bin/activate
   export MUJOCO_GL=egl
   python3 openarm_residual_gate/test_zero_residual_gate_mujoco.py  # 改指向 ReBot 后
   ```

### 推荐下一条工程命令（最小）

```bash
# 从 kinematic 拷出 gate scene 骨架，准备手写 actuator overlay（下一步工程，非本调研执行）
mkdir -p /workspace/openarm_smolvla_residual/sim/rebot_b601_dm
cp /workspace/openarm_smolvla_residual/vendor/rebot_arm_digitaltwin_dm/reBotArm_ros2_DM/src/rebotarm_mujoco/models/rebotarm_b601_kinematic.xml \
   /workspace/openarm_smolvla_residual/sim/rebot_b601_dm/rebotarm_b601_kinematic_upstream.xml
```

然后在同目录新增 `scene_gate.xml`（include + actuators + home），并加 `contract_rebot_b601_dm.json`。

---

## 11. 成功标准核对

| 标准 | 状态 |
|------|------|
| 报告含 DoF/action space | ✅ §3（7 DoF 建议） |
| sim assets 路径或 clone URL | ✅ §4 + DigitalTwin_DM URL；本地 `vendor/rebot_arm_digitaltwin_dm` |
| camera names | ✅ `overhead_rgb`；腕部仅硬件/LeRobot；≠ head |
| known gaps | ✅ §8 |
| recommended M0 switch steps | ✅ §10 |
| 未破坏 OpenArm 工作代码 / 未删 vendor | ✅ |
| 无长训 / 无 PR | ✅ |

---

## 12. 来源速查（本报告引用）

- https://github.com/Seeed-Projects/reBot-DevArm  
- https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM  
- https://wiki.seeedstudio.com/rebot_arm_b601_dm_web_simulator_developer_guide/  
- https://wiki.seeedstudio.com/rebot_arm_b601_dm_ros2_integration/  
- https://wiki.seeedstudio.com/rebot_arm_b601_dm_lerobot/  
- https://wiki.seeedstudio.com/rebot_arm_b601_dm_isaacsim/  
- https://wiki.seeedstudio.com/rebot_arm_b601_rs_web_simulator_developer_guide/ （RS，对照用）  
- https://github.com/Seeed-Projects/reBotArmController_ROS2  
- https://github.com/Seeed-Projects/reBotArm_control_py  
- 本地：`docs/Camera_Layout_Survey.md`、`openarm_residual_gate/contract.json`

