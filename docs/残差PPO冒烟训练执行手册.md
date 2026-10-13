# 残差 PPO 冒烟训练执行手册（PickCubeReBot-v1，冻结 SmolVLA + 7 维残差）

负责人：Residual RL 研究员　日期：2026-10-13　执行者：用户本地执行助手（ducks-OMEN，RTX 5070 Laptop，cu128 / sm_120）
工程路径：`/home/ducks/openarm_smolvla_residual`。**本手册只做冒烟和短预算：冒烟通过之前禁止长训练；冒烟通过之后，长训练也要用户另行批准。** 不开 PR，不改环境文件和冻结的 contract，不照抄 mothinf/vla-residual-rl 的超参和成功率。

## 0. 目标
1. 实现并通过**零残差等价门**：δ≡0 时，残差环境的逐步 `a_exec` 和纯 base 回放完全一致，成功率也一致。
2. 用很小的步数跑通 PPO 的一次完整「采样→更新→再采样」循环，全程没有 NaN，日志字段齐全，安全规则生效。
3. 产出标定数据（δ 分布、clip 命中率、关节限位余量、a_base 统计），给用户定 clip_c 和 λ 用。**本阶段不追求成功率提升。**

## 1. 已核实的设计约束（来源：contract 残差段和 residual fragment）
来自 `openarm_residual_gate/contract_maniskill_rebot_b601_dm.json`（md5 `ece7a861022a84721522c0ec44f1f930`）的 `residual` 段，以及 `contract_maniskill_rebot_b601_dm.residual_fragment.json`（md5 `97a18f6e808983885d1a222caf3d7c45`），状态都是 `draft_cpu_only_no_training`：
- `a_exec = clip(a_base + δ)`，`action_dim = 7`。
- δ 作用于 joint1..6（rad）和 `finger_left`（m）。`finger_right` 是 mimic（= −1×finger_left，见 `rebot_b601_agent.py`），**不加残差**。
- 关节 clip_c 候选 `[0.03, 0.05, 0.10]` rad（schedule）。夹爪 clip_c 是 `null` / `TBD_after_smoke`。
- `last_layer_init = zeros`；`zero_residual_gate = "delta=0 must equal base exactly"`。
- Actor 观测 `[q, qdot, a_base, phase]`，`actor_images=false`，`privileged_state = critic_and_reward_only`。
- 奖励 `ms3_rebot_v0_sparse_success_plus_delta_l2` = `evaluate()['success']` − λ‖δ‖²，`lambda_delta_l2 = null`。
- 门禁：`B0_required_before_residual_training = true`。
- fragment 的 notes 写明：null 字段只能在用户批准的冒烟之后标定，任何数值都不得照抄 mothinf/vla-residual-rl。

其他事实：
- URDF 关节限位（`sim/maniskill_rebot/ReBot_Arm_DM_ms3.urdf`）：j1 ±2.8；j2、j3 ∈ [−3.14, 0]；j4 ∈ [−1.87, 1.57]；j5 ±1.57；j6 ±3.14；finger_left ∈ [0, 0.05] m（effort 8）。
- **限位余量的已知问题（box 上用 `data/mp_demos_n100` 的 diag `track[:, :6]` 算出）**：100 条示范里，各关节到限位的最小余量（rad）是 j1 2.614、j2 0.791、j3 0.878、**j4 0.051**、j5 1.569、**j6 0.04**。另外 `rest` 关键帧 j6=3.10，距上限 3.14 只有 0.04 rad。也就是说，**按"余量 ≥0.1 rad"的规则，j4 和 j6 在示范里、甚至在 reset 时就不满足**。这条规则怎么落地需要用户拍板（D6）；在那之前，冒烟只**记录**余量，不靠它终止回合。
- **夹爪夹持力**：任务说明给的数字是"示范合爪后，夹持力贴 8 N 上限的时间占比中位数 34%"。我在 box 上用 `mp_demos_n100` 的 diag 重算，**没能复现 34%**：取决于定义（单指还是双指、阈值是 7.6、7.9 还是 8.0 N），中位数在 1.4% 到 60% 之间。所以这个数字的出处和口径**待复核**。不管哪种口径，都说明夹爪经常顶在力上限，夹爪的 clip_c 应该小，而且要单独定。

## 2. 工程现状：哪些代码已有，哪些要新写
**已有**（可以参考，但都不能直接用于 ManiSkill）：
- `openarm_residual_gate/residual_env_rebot_mujoco.py`：MuJoCo 版的 ReBot ResidualEnv。7 维控制，actor 观测是 `[q,qdot,a_base,phase]`，有 `step(action, force_delta_zero)`、`roll_a_exec(...)`、`_check_unsafe()`、`_log_fields(...)`。它绑定的是 MuJoCo 和 `contract_rebot_b601_dm.json`，**不是** ManiSkill 的 PickCubeReBot-v1。
- `openarm_residual_gate/test_zero_residual_gate*.py`、`fake_base.py`、`obs_keys_hash.py`：零残差门的写法和 obs_keys 哈希，可以复用。
- `scripts/smolvla_base_rebot.py`（2026-10-13 新增，md5 `930931828bd4b0e6d1304d15d0fb16b3`）：冻结 SmolVLA base，提供 `reset()`、`act(obs)`、`chunk_idx`、`phase`、`last_latency_ms`。见《B0基线评测执行手册》1.1 节。
- `rl/`、`tests/` 目录现在是**空的**。**工程里没有 ManiSkill 版的 ResidualEnv，也没有 PPO 代码。**

**需要新写的文件和接口**（由我们团队实现后推到 OMEN；执行助手不要自己从零写算法，只照本手册跑和验收）：

| 文件 | 接口（最低要求） |
|---|---|
| `rl/residual_env_ms3_rebot.py` | `class ReBotResidualEnvMS3(base, clip_c_joint, clip_c_gripper, lam, seed)`；`reset(seed) -> (actor_obs, info)`；`step(delta, force_delta_zero=False) -> (actor_obs, reward, term, trunc, info)`。内部流程：`a_base = base.act(obs)` → `δ_c = clip(δ, ±c)`（7 维，finger_right 不在动作里，由 mimic 控制器带动）→ `a_exec = clip(a_base + δ_c, URDF 下限, URDF 上限)` → `env.step(a_exec)`。`actor_obs = concat(q[:7], qdot[:7], a_base, phase_feat)`；`info["privileged"]` 放方块位姿和接触力，**只给 critic**。|
| `rl/ppo_residual.py` | Actor MLP 的最后一层权重和偏置**全零初始化**；高斯策略 log_std 的初值**待定**，但要满足"探索噪声 ≤ 当前 c"（讨论稿 6.2）；critic 可以拼接特权状态。CLI：`--total-env-steps`、`--clip-c-joint`、`--clip-c-gripper`、`--lam`、`--seed`、`--ckpt`、`--out`。不设任何默认超参，没传就报错退出 |
| `rl/log_schema.py` | 每步字段：`t, seed, phase, chunk_idx, a_base, delta(原始), delta_c(裁剪后), a_exec, clip_hit(7 维 bool), joint_margin(6), reward, success, q, qdot, latency_vla_ms, obs_keys_hash, reward_version, clip_c` |
| `tests/test_zero_residual_gate_ms3.py` | 第 3 节步骤 1 的门 |
| `tests/test_actor_no_privileged.py` | 断言 actor 的 forward 输入维度 = 7+7+7+phase 维度，而且不含 `obj_pose`、`tcp_pose`、接触力 |

phase 特征的编码（标量 `phase/N`，还是 one-hot，或者 sin/cos）**待定**，归一化统计量来源也待定，都写进 obs_spec 和 obs_keys_hash。

## 3. 前置条件
1. 《统一仿真环境与冒烟测试执行手册》全部通过（`.venv-lerobot` + mani_skill 3.0.1 `--no-deps`，numpy ≥2）。
2. 《LeRobot数据转换与SmolVLA微调执行手册》已产出正式 ckpt 和 `训练记录.json`。
3. 《B0基线评测执行手册》已完成判定评测，**结论是"进入残差冒烟"**（Wilson 区间不满足 `hi<p_lo` 也不满足 `lo>p_hi`），B0 日志已经落盘。没有 B0 结论，不准跑本手册第 3 节步骤 3 及以后的步骤。
4. 第 2 节表里的新文件都已到位，md5 与我们交付时给的一致（交付时补上，现在**待定**）。
5. 关于 base 推理的成本：残差回合中每 N 个控制步要推理一次 SmolVLA（N = ckpt 的 n_action_steps）。每步耗时以 B0 手册第 2 步实测为准。

## 4. 逐步命令
```bash
cd /home/ducks/openarm_smolvla_residual && source .venv-lerobot/bin/activate
CKPT=<B0 用的同一个 ckpt 目录>
mkdir -p logs/residual_smoke
md5sum rl/*.py tests/test_zero_residual_gate_ms3.py tests/test_actor_no_privileged.py scripts/smolvla_base_rebot.py sim/maniskill_rebot/envs/pick_cube_rebot.py openarm_residual_gate/contract_maniskill_rebot_b601_dm.json | tee logs/residual_smoke/md5.txt
```

### 步骤 1：零残差等价门（必须第一个通过）
```bash
python tests/test_zero_residual_gate_ms3.py --ckpt $CKPT --seeds 10000 10001 --out logs/residual_smoke/gate 2>&1 | tee logs/residual_smoke/gate.log
```
测试内容：同一个 seed 跑两遍。A 用 `eval_b0_rebot.py` 同样的纯 base 回放；B 用 `ReBotResidualEnvMS3`，δ 来自零初始化的 actor（也就是网络的真实输出），另外再跑一遍 `force_delta_zero=True`。
验收：
- 三条轨迹的逐步 `a_exec` 要**逐元素相等**，`max|Δ| == 0`。这里的比较要在同一设备、同一 dtype 下进行；如果 GPU 物理引擎有非确定性，先把 base 的动作序列存下来再回放，比较动作，而不是比较状态。
- 两种方式每个 seed 的 `success` 和 `ever_success` 要相同，所以成功率也相同。
- 输出 `gate_report.json`，内容是每个 seed 的 `max_abs_diff`、步数、success。
- **不通过就停下，后面的步骤全部不准跑。**

### 步骤 2：Actor 特权隔离与观测规格
```bash
python tests/test_actor_no_privileged.py 2>&1 | tee logs/residual_smoke/priv.log
```
验收：打印 actor 的输入维度和 `obs_keys_hash`，断言通过；图像和特权键都不在 actor 的输入里。

### 步骤 3：PPO 冒烟（很小的预算）
```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > logs/residual_smoke/gpu.csv & G=$!
python rl/ppo_residual.py --ckpt $CKPT --total-env-steps <待用户定，冒烟量级> \
  --clip-c-joint <从 {0.03,0.05,0.10} 中用户选一个> --clip-c-gripper <用户定> --lam <用户定> \
  --seed 0 --out logs/residual_smoke/ppo_s0 2>&1 | tee logs/residual_smoke/ppo_s0.log
kill $G
```
步数只给"能完成至少 2 次 PPO 更新"的量级。具体数字**待测**：先用步骤 1 的单回合耗时估算，例如 300 步一个回合，再由用户定。
验收（逐条）：
1. 进程正常退出；loss、value、log_prob、δ、a_exec **全程没有 NaN/Inf**。
2. 第一次更新之前的回合，δ 全部为 0（最后一层零初始化）；第一次更新之后，δ 开始非零。
3. `|δ_c| ≤ c` 恒成立，按维统计 `clip_hit` 命中率（数值待测，原样上报）。
4. 关节限位余量 `joint_margin` 有逐步记录，并且标出低于 0.1 rad 的步数占比。按第 1 节，j4 和 j6 在 base 本身的轨迹上就可能低于 0.1，所以这一项是**记录，不是验收失败**（D6 决定之前）。
5. `a_exec` 永远不越过 URDF 限位（安全层在 `a_exec` 之后生效）。
6. 记录每个回合的 success；冒烟阶段**不要求成功率提升**。
7. 有日志里的最低字段（见第 2 节 log_schema）和 `obs_keys_hash`、`reward_version`。

### 步骤 4（可选，需用户先同意）：clip_c 对照
同样的小预算，把 `--clip-c-joint` 换成另外一个候选值，再跑一次 seed 0，只比较 clip 命中率和数值稳定性，**不比较成功率**。

## 5. 提前埋好的诊断数据
- 全部 `.log`、`md5.txt`、`gpu.csv`、`pip freeze`、`nvidia-smi`。
- `gate/gate_report.json`，以及 A、B 两条轨迹的逐步 `a_base`、`a_exec`（npz）。
- `ppo_s0/` 下：每步日志（第 2 节字段，npz 或 parquet）、每次更新的统计（policy loss、value loss、熵、KL、log_std、grad norm、δ 按维的均值、标准差、最大值、clip 命中率、joint_margin 最小值）、actor 和 critic 的初始权重与最终权重（用来核对零初始化）、`run_manifest.json`（参数、版本、ckpt 路径、训练记录 md5）。
- 每个回合的 `max_lift`、`max_hold`、首末帧图像，失败分类沿用 B0 手册第 3 节步骤 5 的脚本，这样和 B0 可以直接对比。
- 夹爪诊断：每步的 `contact_force_left/right`、`finger_left` 的 a_base 和 a_exec，统计接近 8 N 的时间占比，同时列出 >7.6、>7.9、≥8.0 N 三种口径，用来回答第 1 节"34% 无法复现"的问题。

## 6. 常见报错与处理
| 现象 | 处理 |
|---|---|
| 零残差门 `max_abs_diff > 0` | 先检查：是不是在 δ=0 时也经过了一个改变数值的 clip，或者做了 dtype 转换（float64↔float32）；base 的 chunk 队列是否在 reset 时清空；两次运行的 seed 和 base.reset 次序是否相同。把 diff 最大的第一步 t 和那一维上报 |
| 门通过了，但成功率不同 | GPU 物理引擎的非确定性：改为回放存下来的动作，比较两遍；仍不同就上报 |
| 第一次更新前 δ 就非零 | 最后一层没有零初始化，或者采样噪声没有关掉。查初始权重 |
| NaN 出现在 log_prob | log_std 下溢或上溢。上报这一项的统计，不要自己改 |
| clip 命中率接近 100% | c 太小，或者 actor 输出发散。原样上报，作为 D1 的依据 |
| a_exec 越界 | 安全层没有接上，**立刻停**下 |
| 显存不足 | SmolVLA 和 PPO 在同一张卡上。先只跑一个环境，记录显存峰值 |
| `obs_keys_hash` 与交付时给的不一致 | 观测拼接顺序被改动，停下 |

## 7. 需要用户拍板的决策点
| # | 决策 | 候选与理由 | 状态 |
|---|---|---|---|
| D1 | 关节 clip_c | contract 候选 {0.03, 0.05, 0.10} rad。讨论稿的意见：0.03 只适合作起点；如果 base 的典型关节误差 > 0.05–0.1 rad，0.03 会把能纠正的范围裁掉。建议冒烟先用一个值，然后根据 B0 失败样本的误差和 clip 命中率再定 | **待用户定** |
| D2 | 夹爪 clip_c（m） | contract 里是 null。应该小：示范中夹爪常顶在 8 N 力上限（34% 这个数待复核，见第 1 节）。finger_left 行程是 0–0.05 m，候选值待冒烟诊断后给出 | **待用户定**（不预填） |
| D3 | λ（‖δ‖² 系数） | contract 里是 null。λ 和稀疏成功 +1 的量级关系，要看实测 δ 量级后才能定 | **待用户定** |
| D4 | 冒烟总步数、并行环境数 | 只要求够 ≥2 次 PPO 更新，具体值等步骤 1 的耗时出来再定 | **待用户定**（待测） |
| D5 | phase 特征的编码 | 标量 `phase/N`、one-hot（N 维）或 sin/cos。标量最省维度；one-hot 的边界最清楚 | 待用户定 |
| D6 | "关节限位余量 ≥ 0.1 rad"怎么执行 | 示范里 j4 最小余量 0.051、j6 0.04，rest 位姿的 j6 余量只有 0.04，按原样执行会让 base 轨迹本身违规。候选：A 只作为记录和告警；B 只约束 δ 不把余量变得更小（即只对 a_base+δ 比 a_base 更靠近限位的方向生效）；C 改 rest 位姿或改任务（会改环境文件，超出本手册范围）。建议 A 或 B | **待用户定** |
| D7 | log_std 初值和噪声上界 | 只有一条硬约束：探索噪声 ≤ 当前 c（讨论稿 6.2） | 待用户定 |
| D8 | 是否保留 PPO 作为主算法 | 讨论稿建议主路径用 SAC/TD3、PPO 作对照。本手册只覆盖 PPO 冒烟 | 待用户确认 |

## 8. 硬性禁止
- 零残差门（步骤 1）和步骤 2 没通过之前，不跑 PPO。
- 冒烟通过之前，以及冒烟通过之后没有得到用户另行批准之前，都不跑长训练。
- 不把特权状态和图像输入 actor；不改 `pick_cube_rebot.py` 和冻结的 contract；不照抄 mothinf/vla-residual-rl 的数值；不开 PR。
