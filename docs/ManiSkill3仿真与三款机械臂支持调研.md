# ManiSkill3 作为主仿真器 — 机器人选型调研（ReBot B601-DM / OpenArm / AgileX Nero）

**日期：** 2026-10-13（Asia/Shanghai） **范围：** 文献/源码调研 + 本机轻量验证（无训练）。未改动任何现有 contract / 代码。
**范式（保持不变）：** `a_exec = a_base + clip(δ)`；`δ=0 ≡ base gate`；Actor obs = `[proprio, a_base, phase]`；无特权状态、Actor 初期不吃图；success 只进 reward/info。
标注：✅ 已核实（附来源/本机命令） ⚠️ 未核实/推断。

---

## 0. 结论

1. **ManiSkill3 已正式发布 v3.0.1（2026-04-21，脱离 beta）**，内置 `so100`、`koch-v1.1`、`widowxai`、`xarm6_*`、`panda` 等，**三款目标机器人都没有内置 agent，也没找到社区 ManiSkill 移植版**（⚠️ 只是没搜到，不代表一定不存在）。
2. **本机（box）没有 NVIDIA GPU，也没有 Vulkan**：`mani_skill` 能 import，agent 注册表也能列出来，但 SAPIEN 在加载 URDF 时就要创建 `RenderMaterial`，所以 `gym.make` 报错 `failed to find a rendering device`。即使设 `render_backend='none'` 也一样。**真正的 smoke 测试必须放到用户的 4090 机器上。**
3. **首选 ReBot B601-DM**：LeRobot 原生支持（`rebot_b601_follower`），关节绝对角（度）7 维动作，和 ManiSkill `pd_joint_pos` 一一对应（只差度↔弧度换算和夹爪映射）。HF 单臂数据最多（≥4 个公开数据集，最大 200 ep），和现有 MuJoCo gate（n=7）也一致。**OpenArm v1 单臂排第二**；**Nero 暂缓**，因为它没有 HF 官方 LeRobot 驱动。

## 1. ManiSkill3 状态

| 项 | 结论 | 来源 |
|---|---|---|
| 版本 | `mani_skill 3.0.1`（PyPI，2026-04-21） | https://github.com/mani-skill/ManiSkill/releases/tag/v3.0.1 , https://pypi.org/project/mani-skill/ |
| Python | 本机 py3.13 上 `pip install` 失败：`mplib==0.1.1` 没有 3.13 wheel。改用 **uv + Python 3.11** venv（`/tmp/ms311`）后安装成功 ✅ | 本机 |
| GPU 并行 | SAPIEN + PhysX GPU，`num_envs` 可以到上千；官方论文宣称带渲染时可达 **30,000+ FPS**（⚠️ 论文标题级数字，对应的硬件和任务没有在本配置下核实） | ManiSkill3 论文（RSS 2025），https://arxiv.org/abs/2410.00425 |
| 无头渲染 | 需要 Vulkan。本机 `nvidia-smi` 不存在 → svulkan2 报 `ErrorIncompatibleDriver`。CPU 回退（`sim_backend='physx_cpu'`）在本机**也跑不起来**，因为加载 URDF 需要渲染设备 ✅ | 本机 |
| 机器人导入 | 继承 `BaseAgent`，设 `uid`、`urdf_path` 或 `mjcf_path`，用 `@register_agent()` 注册，在 `_controller_configs` 中定义控制器。本机源码确认 `base_agent.py` 同时支持 URDF 和 MJCF（`MJCFLoader`） ✅ | https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html ；本机源码 |
| 内置机器人（本机 `REGISTERED_AGENTS`） | panda / panda_wristcam / so100 / koch-v1.1 / widowx250s / widowxai(_wristcam) / xarm6_* / xarm7_ability / ur_10e / xlerobot / unitree_* 等。**没有 rebot / openarm / nero** ✅ | 本机 |
| 控制器 | `pd_joint_pos`、`pd_joint_delta_pos`、`pd_joint_target_delta_pos`、`pd_ee_delta_pos`、`pd_ee_delta_pose` 等。so100 只实现了 joint 类控制器 | https://maniskill.readthedocs.io/en/latest/robots/so100/index.html |
| 抓取任务 | `PickCube-v1`（另有 `PickCubeSO100-v1`、`PickCubeWidowXAI-v1`）。`evaluate()` 返回 `"success": is_obj_placed & is_robot_static`，成功时 reward 置为 5 ✅（本机源码 `pick_cube.py:155,185`）。新机器人需要加入 `SUPPORTED_ROBOTS`，或像 SO100 那样写一个子类任务 | 本机源码 |
| 相机 | `CameraConfig(uid, pose, w, h, fov, mount=link)`，可挂到 agent link 上，用来定义 wrist 相机 | 官方文档 sensors |
| LeRobot / sim2real | 有 SO100 / Koch 低成本臂 + `xlerobot` 的内置 agent；ManiSkill 官方 LeRobot sim2real 示例只针对 SO100（⚠️ 具体示例路径未复核） | 文档 |
| RL baseline | `examples/baselines/ppo/ppo.py`（GPU 并行 PPO） | https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo |
| Residual RL 示例 | 官方仓库里**没找到** residual RL 示例 ⚠️。需要自己写 wrapper | — |

## 2. 三款机器人对比（含 steering 第 1–8 项）

| # | 项 | ReBot B601-DM | OpenArm（v1 单臂 / v2 双臂） | AgileX Nero |
|---|---|---|---|---|
| — | 是什么 | 6-DoF + 夹爪，达妙电机（见 ReBot_B601_DM_Survey.md） | 7-DoF + 夹爪 / 臂，双臂版有 `bi_openarm_follower` | **7-DoF 臂**，自研电机，负载 3 kg，臂展 580 mm，重复定位 0.1 mm，重 4.8 kg，**售价 US$2,499**（https://global.agilex.ai/products/nero） |
| — | ManiSkill 内置 / 社区移植 | 无 / 未找到 | 无 / 未找到 | 无 / 未找到（官方移植目标是 Isaac Lab） |
| — | LeRobot 类 | **原生** `rebot_b601_follower`（`motor_family=dm`）+ `bi_rebot_b601_follower`（lerobot HEAD `24f15b9`，`src/lerobot/robots/`）✅。注：部分 HF 数据集记录的类型是 `seeed_b601_dm_follower`（旧名或插件名，⚠️） | **原生** `openarm_follower` / `bi_openarm_follower` ✅ | **不在** HF lerobot 主仓，也不在官方第三方插件列表（那里只有 Piper）✅。AgileX 自己维护 fork https://github.com/agilexrobotics/lerobot-agilex （走 ROS2 + data_tools 流程，不是标准 `robot.type`） |
| 1 | `action` / `observation.state` | 7 维：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper` 的 `.pos`，**绝对关节角，单位度**。夹爪也在“公共关节度”空间（action 是绝对目标）✅（源码 `rebot_b601_follower.py:286,316`；数据集卡 binhpham/portal-rebot-pick-microcontroller） | 单臂 8 维：`joint_1..7, gripper` 的 `.pos`，**度**（`MotorNormMode.DEGREES`）。夹爪默认范围 `(-65, 0)` 度 ✅（`config_openarm_follower.py:31`）；可选 `.vel/.torque` 观测 | ⚠️ 未核实。pyAgxArm 的 `move_j` 用 7 关节**弧度**；夹爪单位没查到。lerobot-agilex 数据的维度未核实 |
| 2 | HF 单臂数据规模 | LeWAM/b601_pusht_pick_and_place **200 ep**（3 相机）；binhpham/portal-rebot-pick-microcontroller 50 ep / 27,502 帧；SieloRobotics/rebot_pick_place_20261005_170722 15 ep / 26,868 帧；adrfm/pick_place_b601_v2 10 ep | openrob-furg/openarm-single-pick-place-lego **200 ep / 47,787 帧**（右臂，用来微调 SmolVLA）；另有 enactic 双臂 / 仿真 cell 数据（见 Camera_Layout_Survey.md） | 未找到公开 LeRobot 格式 Nero 数据 ⚠️ |
| 3 | ManiSkill 控制器能否直接输出同一动作空间 | 能：`pd_joint_pos`（rad）。需要一个**薄 ActionAdapter**，只做 deg↔rad 换算 + 夹爪映射（公共度 ↔ URDF 关节量） | 能：同上，8 维。夹爪 `(-65,0)` 度要映射到 URDF 的指间距（m）或关节量 | 关节部分能。夹爪映射未知，需要 Adapter |
| 4 | 现成抓取任务 + `success` | `PickCube-v1` 有 `success`；需要注册新 agent 或写 `PickCubeReBot-v1` 子类（参照 `PickCubeSO100-v1`），并调 cube 尺寸和工作空间 | 同左 | 同左 |
| 5 | 与冻结 SmolVLA 同空间的模式 | **`pd_joint_pos`**：SmolVLA 输出绝对关节目标，所以 `a_exec = a_base + clip(δ)` 直接在绝对关节空间相加。`pd_joint_delta_pos` 会改变 `a_base` 的语义；`pd_ee_delta_pose` 和 LeRobot 实机的关节空间不一致，**不选** | `pd_joint_pos` | `pd_joint_pos` |
| 6 | GPU 并行数 / 步时 | 只有论文级数字（上表）。**本机为零**；本文只提供 CPU 端接口草案，实测要在 4090 上做（建议记录 num_envs ∈ {1,64,512}，state 模式的 steps/s） | 同左 | 同左 |
| 7 | URDF/MJCF 能否直载、夹爪 | 有官方 URDF + MJCF（MJCF `nu=0`，见旧调研）。ManiSkill 会忽略 MJCF actuator，由控制器驱动，所以 `nu=0` 在这里**不是阻塞问题**。夹爪是否是 mimic 关节 ⚠️ 未核实；如果是双指 mimic，就照 ManiSkill panda 的做法用 `PDJointPosMimicController` 或单关节驱动 | 有官方 URDF（openarm_description xacro → 需要展开）和 MJCF（vendor/openarm_mujoco）。v1 夹爪有两个手指关节（⚠️ 是否 mimic 待查），可能需要 mimic 控制器 | URDF 在 https://github.com/agilexrobotics/agx_arm_urdf （`nero/urdf/nero_description.urdf` + `nero_with_gripper_description.xacro`）。需要合并 xacro，并把 `package://` 改成相对路径（Isaac 移植教程已经这样做过）。夹爪关节是 `gripper_joint1/2`（⚠️ 是否 mimic 待查）。许可证 ⚠️ 未核实 |
| 8 | 官方腕部相机 | 硬件有 D405 腕部支架 STEP（reBot-DevArm/hardware/camera-mounts），**没有仿真外参** → **外参暂不编写（unauthored）** | v2 MJCF 有 `camera_wrist_*`（`ee_base_link`），v1 没有。硬件 D405 wrist 支架见 issue #307 → v1 **外参暂不编写** | 没找到官方腕部相机仿真定义 → **外参暂不编写** |

## 3. 推荐与 M0 计划（ManiSkill3）

**先做 ReBot B601-DM 单臂；OpenArm v1 单臂第二；Nero 暂缓**，等官方 LeRobot 驱动或数据出现后再上。

M0 步骤：
1. **环境**：在 4090 机器上建 Python 3.11 venv，`pip install mani_skill torch`，再用 `python -m mani_skill.examples.demo_random_action -e PickCube-v1` 验证 Vulkan。
2. **agent 移植**：`agents/rebot_b601.py` 里写 `@register_agent() class ReBotB601DM(BaseAgent)`，用 `uid="rebot_b601_dm"` 和官方 URDF（优先 URDF；MJCF 作备选）。添加 keyframe `rest`（对齐 FakeBase home）；控制器只定义 `pd_joint_pos`（arm 6 + gripper 1）。PD 增益从 so100 的值起步，并在文档里标为“未辨识”。
3. **碰撞**：检查 collision mesh 是否为凸包（ManiSkill 会自动做凸分解，但细指需要人工检查）。
4. **任务**：参照 `PickCubeSO100-v1` 写 `PickCubeReBot-v1`，复用 `evaluate()` 的 `success`。
5. **ActionAdapter**：LeRobot 字典（度）↔ ManiSkill 向量（rad）。顺序固定为契约里的 7 个关节名。
6. **Residual wrapper**：`a_exec = a_base + clip(δ, ±b)`。Actor obs 用 `[qpos(7), a_base(7), phase]`；success 只放进 info。
7. **相机键**：第一阶段只放一个世界固定相机，键名用 `overhead`（对齐旧调研），wrist 暂不定义。
8. **门禁测试**：`δ=0` 时，ManiSkill 下的 base 轨迹应当与 FakeBase 一致；动作 shape 是 7；obs 不含物体位姿。新建 `contract_maniskill_rebot_b601_dm.json`（**不改**现有契约）。

## 4. 风险

| 风险 | 说明 |
|---|---|
| 无 GPU / Vulkan | 本机无法实测，所有 ManiSkill 运行都要用 4090 机器 |
| URDF 质量 | 惯量、关节限位、mesh 路径（`package://`）、`.dae` 材质；Nero 需要合并 xacro |
| 夹爪 mimic | SAPIEN 不会自动处理 URDF `<mimic>`，需要 mimic 控制器或手工耦合 |
| 动作空间错配 | 实机是度 + 夹爪“公共度”，仿真是 rad + 指间距；Adapter 必须测往返一致性 |
| PD 增益 / 动力学 | 内置 quality 等级里，新 agent 只到 “B/未辨识”，sim2real 有差距 |
| Nero LeRobot 支持 | 用户认为“三款都有良好 LeRobot 支持”，但对 Nero 来说**不成立**：只有 AgileX fork + ROS2 流程 |
| 名称漂移 | `seeed_b601_dm_follower` vs `rebot_b601_follower`，加载数据集时需要做别名映射 |

## 5. 本机验证记录

- `nvidia-smi: command not found`；py3.13 安装失败（mplib）；`uv venv -p 3.11 /tmp/ms311` 安装成功，`mani_skill.__version__ == 3.0.1`。
- `REGISTERED_AGENTS` 已列出（见 §1）。
- `gym.make('PickCube-v1', sim_backend='physx_cpu', render_backend='none', robot_uids='so100')` 报 `RuntimeError: failed to find a rendering device`（在 sapien `urdf_loader._build_link`）。尝试 apt 装 mesa lavapipe 时被中断，没有完成。
