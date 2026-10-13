# LeRobot 数据转换与 SmolVLA 微调执行手册（ReBot B601-DM 抓方块）

> 编写：VLA与数据研究员，2026-10-13。面向执行者（用户本地助手），按顺序执行，每步核对验收标准，不通过就停下按「常见报错」处理或上报。
> 来源：lerobot 0.6.1 wheel 源码（`lerobot/datasets/lerobot_dataset.py`、`policies/smolvla/configuration_smolvla.py`、`configs/train.py`、`processor/normalize_processor.py`）；`lerobot/smolvla_base` 的 HF `config.json`；本工程 `data/mp_demos_n100` 实测统计。文中没有实测依据的地方都标了「待实测」，请不要自行填数。

## 0. 目标
1. 把对齐手腕朝向后的 100 条运动规划示范 `data/mp_demos_yaw_n100` 转成 LeRobot 数据集，得到正式 `stats_id`。
2. 先跑一次短的微调冒烟，测出显存和每步耗时；用户确认步数和 batch 后，再从 `lerobot/smolvla_base` 正式微调。
3. 产出给 B0 评测用的 checkpoint，交给 Residual RL研究员（见 `docs/B0基线评测执行手册.md`）。

**硬性规矩**：冒烟没通过、用户没确认之前，不启动正式训练。示范种子是 0–99，评测种子从 10000 起，任何种子 ≥10000 的数据都不能进训练集。

## 1. 前置条件
| 项 | 要求 | 核对方式 |
|---|---|---|
| 机器 | ducks-OMEN，RTX 5070 Laptop（sm_120） | `nvidia-smi` |
| 工程目录 | `/home/ducks/openarm_smolvla_residual` | |
| 采集环境 | `.venv-ms3`（Python 3.11、numpy 1.26.4、mplib 0.1.1），**本手册不用它，也不要往里装 lerobot** | |
| 训练环境 | `.venv-lerobot`：Python 3.12、`lerobot[smolvla,dataset]==0.6.1`、torch cu128 | 见 1.1 |
| 系统依赖 | `ffmpeg`（视频编码用） | `ffmpeg -version` |
| 输入数据 | `data/mp_demos_yaw_n100/`，`meta.json` 里 `align_yaw: true`，计数 ok=100 | 见 2.1 |
| 脚本版本 | `scripts/转换LeRobot数据集_rebot.py` md5 `ef57317a886d0f214fc73c68bdbe0b6b`；`scripts/gen_mp_demos_rebot.py` md5 `cfcc9fc4fc437bce4fae175eb2fc4675` | `md5sum` |
| 网络 | 能访问 Hugging Face（下载 `lerobot/smolvla_base` 和 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`） | |

### 1.1 建训练环境（还没建的话）
```bash
cd /home/ducks/openarm_smolvla_residual
uv venv --python 3.12 .venv-lerobot
source .venv-lerobot/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "lerobot[smolvla,dataset]==0.6.1"
python -c "import lerobot,torch,numpy;print(lerobot.__version__,torch.__version__,numpy.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability())"
```
**验收**：打印 `0.6.1`，torch 在 2.7–2.11 之间且带 `+cu128`，numpy ≥2.0，`True`，`(12, 0)`。
注意：lerobot 0.6.1 要求 torch <2.12、numpy 在 2.0 到 2.3 之间。如果 uv 要把 torch 换成不带 cu128 的版本，先停下来。

## 2. 数据转换

### 2.1 核对输入
```bash
grep -o '"align_yaw": [a-z]*' data/mp_demos_yaw_n100/meta.json
grep -o '"counts": {[^}]*}' data/mp_demos_yaw_n100/meta.json
ls data/mp_demos_yaw_n100/episode_*.npz | wc -l
grep -ho '"grasp_yaw": [-0-9.e]*' data/mp_demos_yaw_n100/plans_000*.json | head
```
**验收**：`align_yaw: true`；ok=100；正好 100 个 episode 文件；`grasp_yaw` 是 [-0.785, 0.785] 范围内各不相同的数。不满足就停下，说明数据是旧脚本采的。

### 2.2 先试转 3 条（检查接口、估算耗时）
```bash
source .venv-lerobot/bin/activate
mkdir -p /tmp/lr_try && cp data/mp_demos_yaw_n100/meta.json data/mp_demos_yaw_n100/episode_0000[0-2].npz /tmp/lr_try/
time python scripts/转换LeRobot数据集_rebot.py --src /tmp/lr_try --out data/lerobot/试转3条 --repo-id local/rebot_try3 --std-floor 1e-3
```
**预期输出**：先打印 `stats_id = rebot_b601_mp_n3_… | 帧数 … | std 下限生效维度 [...]`，接着逐条打印 `[i/3] episode_… N 帧 已用 …s`，然后是 `已对 meta/stats.json … 施加 std 下限`，最后是 `完成:`。
**验收**：`data/lerobot/试转3条/meta/info.json` 存在；记下 `time` 的 real 时间，全量耗时大约是它的 100/3 倍。

### 2.3 全量转换
```bash
time python scripts/转换LeRobot数据集_rebot.py --src data/mp_demos_yaw_n100 \
  --out data/lerobot/rebot_b601_pickcube_mp_yaw_n100 \
  --repo-id local/rebot_b601_pickcube_mp_yaw_n100 --std-floor 1e-3 2>&1 | tee logs/转换_yaw_n100.log
```
**验收**（逐条核对）：
1. 日志打印的 `stats_id` 形如 `rebot_b601_mp_n100_<16 位十六进制>`，**把它记下来上报**。它会取代旧数据的 `…091a05ee19fea9ba`，后者作废。
2. 日志里 `std 下限生效维度` 预期包含 `joint5`。这是旧数据上测到的（joint5 动作 std 只有 0.0003 rad）；新数据要以实际打印为准。
3. 用下面的脚本核对数据集：
```bash
python - <<'PY'
import json
r="data/lerobot/rebot_b601_pickcube_mp_yaw_n100"
info=json.load(open(f"{r}/meta/info.json")); st=json.load(open(f"{r}/meta/stats.json"))
print("episodes",info.get("total_episodes"),"frames",info.get("total_frames"),"fps",info.get("fps"))
print("features",sorted(info["features"].keys()))
print("action std",[round(x,5) for x in st["action"]["std"]])
PY
```
- episodes = 100；frames 等于 100 条示范步数之和（旧数据是 18140，新数据以实际为准）；fps = 20。
- features 至少包含 `action`、`observation.state`、`observation.images.wrist`、`observation.images.overhead`、`task_index`。
- action std 里每一维都 ≥ 0.001。

4. 抽一条回放肉眼检查：`lerobot-dataset-viz --repo-id local/rebot_b601_pickcube_mp_yaw_n100 --root data/lerobot/rebot_b601_pickcube_mp_yaw_n100 --episode-index 0`。看两路相机画面正常、方块被夹起、动作曲线连续。

**为什么要设 std 下限**：lerobot 0.6.1 的 SmolVLA 对 state 和 action 用 MEAN_STD 归一化，分母是 std + 1e-8（`normalize_processor.py`）。joint5 的 std 只有 0.0003 rad 时，归一化会把噪声放大大约 3300 倍。转换脚本已经把下限同时写进训练实际读取的 `meta/stats.json`。

## 3. 微调设计（依据与待定项）
| 项 | 取值 | 依据 / 状态 |
|---|---|---|
| 初始化 | `--policy.path=lerobot/smolvla_base` | 官方示例（`modeling_smolvla.py` 文档串） |
| 相机键映射 | wrist → `camera1`，overhead → `camera2` | smolvla_base 的 config 输入是 camera1/2/3，256×256 |
| camera3 | 缺一个相机：先不补，报错再加 `--policy.empty_cameras=1` | 待实测 |
| state / action 维度 | 7（smolvla_base 是 6），都会 pad 到 `max_state_dim/max_action_dim=32` | 能否直接加载待冒烟验证 |
| chunk_size / n_action_steps | 50 / 50（默认值，和 ChunkScheduler 一致） | 50 步 = 2.5 s（20 Hz） |
| 冻结 | `freeze_vision_encoder=true`、`train_expert_only=true`（默认） | **用户拍板**，见第 6 节 |
| 学习率 / warmup | 1e-4 / 1000（默认） | 默认值 |
| steps、batch_size | **待冒烟实测后定** | 不预填 |
| seed | 1000（默认），记进日志 | |
| 训练集 | 只用 `mp_demos_yaw_n100`（种子 0–99） | 防止测试集泄漏 |

## 4. 微调冒烟（短跑，测显存和速度）
```bash
source .venv-lerobot/bin/activate
mkdir -p logs
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > logs/冒烟_gpu.csv &
GPU_LOG=$!
time lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/rebot_b601_pickcube_mp_yaw_n100 \
  --dataset.root=data/lerobot/rebot_b601_pickcube_mp_yaw_n100 \
  --rename_map='{"observation.images.wrist":"observation.images.camera1","observation.images.overhead":"observation.images.camera2"}' \
  --batch_size=8 --steps=300 --log_freq=20 --save_freq=300 \
  --output_dir=outputs/smolvla_rebot_冒烟 --job_name=smolvla_rebot_smoke \
  --policy.push_to_hub=false --wandb.enable=false \
  2>&1 | tee logs/冒烟_train.log
kill $GPU_LOG
```
**验收**：
1. 能正常跑完 300 步，loss 没有 NaN，整体呈下降趋势（日志里每 20 步一条）。
2. 记录以下数字上报：每步耗时（日志里的 `updt_s` 或用总时间除以步数）、显存峰值（`logs/冒烟_gpu.csv` 中 memory.used 的最大值）。
3. `outputs/smolvla_rebot_冒烟/checkpoints/` 下有 300 步的 checkpoint。

**全量耗时估算**：正式步数 × 冒烟实测的每步耗时，再加大约 10% 的保存开销。这只是估算，以实测为准。

## 5. 正式微调（用户确认后才执行）
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/rebot_b601_pickcube_mp_yaw_n100 \
  --dataset.root=data/lerobot/rebot_b601_pickcube_mp_yaw_n100 \
  --rename_map='{"observation.images.wrist":"observation.images.camera1","observation.images.overhead":"observation.images.camera2"}' \
  --batch_size=<用户定> --steps=<用户定> --save_freq=<steps/5> --log_freq=100 \
  --output_dir=outputs/smolvla_rebot_yaw_n100_s<steps> --job_name=smolvla_rebot_v0 \
  --policy.push_to_hub=false --wandb.enable=false \
  2>&1 | tee logs/正式微调_s<steps>.log
```
训练完成后，把以下内容写进 `outputs/.../训练记录.json`：stats_id、数据目录、脚本 md5、lerobot/torch 版本、steps、batch、seed、总耗时、最终 loss。这份记录就是交给 B0 的冻结 base 的版本凭据。

## 6. 需要用户拍板
1. **steps 和 batch_size**：等第 4 步测出每步耗时和显存后，给出 2–3 组候选和对应的估算耗时，由用户选。
2. **是否只训动作专家**：默认 `train_expert_only=true`，显存小、速度快，是官方默认。另一个选项是同时训 VLM 文本层（`--policy.train_expert_only=false`），显存和耗时都会上升，在 5070 Laptop 上可能放不下。建议先用默认。
3. **camera3 是否补空相机**：只有冒烟报特征不匹配时才需要决定。

## 7. 常见报错
| 现象 | 处理 |
|---|---|
| `Feature mismatch ... Missing features: observation.images.camera3` | 加 `--policy.empty_cameras=1` 重跑冒烟 |
| 7 维和 6 维不匹配导致加载失败（`size mismatch` 出现在 state_proj 或 action 相关层） | 停下，把完整报错上报；不要改成 6 维、也不要删维度，由 VLA 决定方案 |
| CUDA OOM | batch_size 依次减半（8→4→2）；仍然不行就上报 |
| `no kernel image is available` / sm_120 不支持 | torch 不是 cu128 版本，按 1.1 重装 |
| 转换时报 `add_frame` 缺 `task` 或 feature 校验失败 | 把完整报错上报，脚本会按接口修 |
| 输出目录已存在、脚本退出 | 这是有意为之，防止覆盖；换个目录或人工确认后再删 |
| 下载 smolvla_base 失败 | 检查网络和 HF 镜像（`HF_ENDPOINT`）；必要时先 `huggingface-cli download lerobot/smolvla_base` |
| loss 是 NaN | 检查 `meta/stats.json` 里有没有 std 很小的维度没加上下限；把日志上报 |

## 8. 要提前埋好的诊断数据（每次都留）
- `logs/转换_*.log`、`logs/冒烟_train.log`、`logs/冒烟_gpu.csv`、`logs/正式微调_*.log`。
- 环境清单：`uv pip freeze > logs/环境清单_lerobot.txt`。
- 归一化检查：训练前用 `meta/stats.json` 把全量 action 归一化一遍，记录每一维的 min/max，存到 `logs/归一化范围.json`；任何一维绝对值超过 20 都算异常。
- 训练结束后，用 checkpoint 对训练集的 3 条做开环预测，记录预测动作和示范动作的逐维 MSE，存到 `logs/开环误差.json`。开环误差小不能说明闭环一定能成功，闭环要靠 B0 评测。

## 9. 交付物
- `data/lerobot/rebot_b601_pickcube_mp_yaw_n100/`（含 `统计量_stats.json`、`meta/stats.json`）
- 正式的 `stats_id`（写回 `configs/smolvla_finetune_rebot_v0.yaml`，由 VLA 负责）
- `outputs/smolvla_rebot_yaw_n100_s<steps>/checkpoints/last/` 和 `训练记录.json`
- 第 8 节列出的所有日志
