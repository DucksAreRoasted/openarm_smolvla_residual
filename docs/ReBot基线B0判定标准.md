# B0 判定口径（PickCubeReBot-v1，冻结 SmolVLA，δ=0）

- 成功定义：episode 结束时 `evaluate()["success"]`；另记 `ever_success`。
- 测试 seed 与示范生成 seed 不相交，固定写入 summary。
- 报告：成功数 k/n + Wilson 95% 区间，不只报点估计。
- 判定（阈值 p_lo、p_hi 由用户确认，本文不预设数值）：
  - Wilson 上界 < p_lo：base 太弱，回去改 base（加 demo/微调），不训残差。
  - Wilson 下界 > p_hi：没有改进空间，不训残差。
  - 其余：进入残差冒烟。
- 冒烟 n 只用于跑通流程，不用于判定；判定用的 n 由用户确认。
- 失败样本人工标 failure_mode（未接近/抓空/滑落/超时等）。

## 已确认（2026-10-13，用户确认）
- 成功：lift_height ≥ 1.25×方块边长（5 cm）且 is_grasped，连续保持 k=10 控制步。
- 每回合上限 max_episode_steps=300（示范 177–185 步）。
- 依据：mp_demos_n100 全部 100 条在第 142–152 步过 5 cm，合爪前抬升为 0；seed 92 有一步夹持抖动，k 不宜过大。
- 待复核：B0 出现失败样本后，检查失败轨迹的最大抬升，确认无误判成功。
