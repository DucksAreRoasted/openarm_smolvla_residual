# ReBot 抓方块冒烟测试运行说明（PickCubeReBot-v1，只冒烟，不训练）

运行机器：ducks-OMEN（RTX 5070，Blackwell，sm_120）。环境由用户自行安装。

## 1. 环境（要点）
- PyTorch 必须是 CUDA 12.8 版本（cu128），否则 5070 无法识别：
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- 然后：`pip install "mani_skill==3.0.1" pillow scipy "numpy<2"`（mplib 0.1.1 在 numpy 2.x 下会 segfault，实测 1.26.4 可用）
- 示范规划需要 `sim/maniskill_rebot/plan/` 目录（规划专用 URDF、SRDF 和凸包网格）
- 先确认：
  `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_arch_list())"`，输出里须含 `sm_120`
  `nvidia-smi`、`vulkaninfo --summary`（无则装 vulkan-tools / libvulkan1）

## 2. 运行
```bash
python scripts/check_maniskill_rebot_static.py
python scripts/smoke_pickcube_rebot_4090.py      # 文件名沿用，5070 同样适用
python scripts/eval_b0_rebot.py --policy hold --n-episodes 5   # 只跑通流程，不作 B0 判定，日志应出现在 logs/b0_rebot
python scripts/gen_mp_demos_rebot.py --n 1   # 核 mplib 能否加载 URDF（本机 CPU 曾 segfault）
```

## 3. 通过标准与回传
- `report.json` 中 env_build / cam_wrist / cam_overhead / delta0_equals_base 全为 true（含 cube_fully_in_wrist / cube_fully_in_overhead，初始姿态下方块完整入画），OVERALL PASS。
- report 里新增 tcp_pose_init（p、q）和接近轴（end_link 局部 +x），供核对抓取四元数。
- 回传整个 `logs/maniskill_rebot_smoke/<时间戳>/`（report.json、wrist.png、overhead.png），以及 B0 hold 的输出。
- 请顺带看 report 里的 active_joints 顺序和 action_space，用于核对关节顺序和夹爪 mimic。
