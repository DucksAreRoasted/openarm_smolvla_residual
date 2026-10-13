#!/usr/bin/env python3
"""ReBot B601-DM PickCube 运动规划示范生成（方案一 base 来源）。

状态：草稿，未在 GPU 上跑过。本机无 NVIDIA/Vulkan，只做过 import/语法检查。
依赖：.venv-ms3（mani_skill==3.0.1, mplib==0.1.1, numpy<2 —— numpy 2.x 下 mplib 0.1.1 加载任何 URDF 都会 segfault）；env `PickCubeReBot-v1` 由仿真侧提供。

录制口径（与 camera_config_v0.json 一致）：
- 每个 env.step = 一个控制步 = 20 Hz（MS3 sim 100 Hz / control 20 Hz）；mplib time_step=1/20。
- 每控制步取一帧 wrist + overhead；30 fps 原始流不在此脚本录制。
- action = [joint1..6 (rad, 绝对), finger_left (m, 绝对)] 7 维；与 pd_joint_pos 同空间。
- success 只取 env.evaluate()["success"]，不自造阈值；只保存 success=True 的 episode（另记失败数）。
- 相机位姿随机化：--cam-rand 开关已留，范围读 camera_config_v0.json，范围为 null 时拒绝开启。

用法（4090）：
  MUJOCO_GL=egl .venv-ms3/bin/python scripts/gen_mp_demos_rebot.py --n 5 --out data/mp_demos_smoke --seed 0
冒烟通过前不要放大 --n。
"""
import sys as _s, pathlib as _p
_R=_p.Path(__file__).resolve().parents[1]
_s.path.insert(0,str(_R)); _s.path.insert(0,str(_R/"sim/maniskill_rebot/envs"))
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAM_CFG = ROOT / "sim/maniskill_cameras/camera_config_v0.json"
# 规划专用 URDF：只含碰撞几何 + 预生成 .convex.stl（mplib 0.1.1 强制 convex=True），配 SRDF
URDF = ROOT / "sim/maniskill_rebot/plan/ReBot_Arm_DM_ms3_plan.urdf"
SRDF = ROOT / "sim/maniskill_rebot/plan/ReBot_Arm_DM_ms3_plan.srdf"
CONTROL_HZ = 20
GRIP_OPEN, GRIP_CLOSE = 0.05, 0.0  # finger_left, m, URDF 限位
# 示范轨迹参数（冒烟用，按首条成功轨迹再调；不是标定值）
GRASP_DEPTH_FRAC = 0.5   # 指尖压到方块中心以下 0.5*half
GRASP_MIN_Z = 0.008      # 指尖离地下限 m
SETTLE_STEPS = 5
CLOSE_STEPS = 20         # 旧版 10
HOLD_STEPS = 15
LIFT_DZ = 0.15
GRASP_Q = [0.7071, 0.0, 0.7071, 0.0]  # wxyz
ALIGN_YAW = True  # --no-align-yaw 可复现旧的固定抓取朝向（mp_demos_n100）
DIAG_KEYS = ["lift_height", "is_grasped", "contact_force_left", "contact_force_right", "hold_count", "success"]

def load_cam_rand():
    r = json.loads(CAM_CFG.read_text())["residual_camera_randomization"]
    if r["pos_range_m"] is None or r["rot_range_rad"] is None:
        sys.exit("[gen] 相机随机化范围未定（camera_config_v0.json 中为 null），不能开 --cam-rand")
    return r

def to_np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)

def make_planner():
    import mplib
    sys.path.insert(0, str(ROOT))
    from sim.maniskill_rebot import rebot_b601_agent as A
    return mplib.Planner(urdf=str(URDF), srdf=str(SRDF), move_group=A.TCP_LINK_NAME,
                         joint_vel_limits=np.ones(6), joint_acc_limits=np.ones(6))

def follow(env, plan, grip, rec):
    for q in plan["position"]:
        a = np.concatenate([q[:6], [grip]]).astype(np.float32)
        rec.step(env, a)

def gripper_to(env, q_arm, grip, n, rec):
    for _ in range(n):
        rec.step(env, np.concatenate([q_arm, [grip]]).astype(np.float32))

class Recorder:
    def __init__(self):
        self.frames = []
        self.plans = []   # 每次规划的目标、起点、状态、耗时
        self.track = []   # 每步 实际 qpos(8) 与 目标 action 的误差，查 PD 跟踪/增益问题
        self.diag = []  # 每步诊断，供残差侧标定 lift_h_mult / k，不进训练集
    def step(self, env, a):
        obs, _, _, _, info = env.step(a)
        sd = obs["sensor_data"]
        ev = env.unwrapped.evaluate()
        qp = to_np(env.unwrapped.agent.robot.get_qpos())[0]
        self.track.append(np.concatenate([qp, qp[:7] - a]).astype(np.float32))
        self.diag.append([float(to_np(ev[k]).reshape(-1)[0]) for k in DIAG_KEYS])
        self.frames.append(dict(
            action=a,
            state=to_np(env.unwrapped.agent.robot.get_qpos())[0][:7].astype(np.float32),  # joint1..6 + finger_left
            wrist=to_np(sd["wrist"]["rgb"])[0],
            overhead=to_np(sd["overhead"]["rgb"])[0],
        ))
        return info

def solve(env, planner, rec):
    """抓取：预抓取上方 → 下降 → 合爪 → 抬升。位姿取自 env 特权状态，仅用于生成示范，不进 Actor。"""
    import sapien
    u = env.unwrapped
    qpos = to_np(u.agent.robot.get_qpos())[0]
    obj_p = to_np(u.cube.pose.p)[0]
    # 就绪姿态已改为前倾（非竖直），抓取朝向用竖直朝下：end_link +x 绕 y 转 90°，
    # 已在旧就绪姿态 FK 上核对 = [0.7071,0,0.7071,0]；新就绪姿态下 9 个出生位置规划均通过（CPU）。
    down_q = GRASP_Q
    if ALIGN_YAW:  # 按方块 yaw 对齐手腕：方块 90° 对称，取最近的 90°，偏转限制在 ±45°
        w, x, y, z = to_np(u.cube.pose.q)[0]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        a, b = qz, np.array(GRASP_Q)  # 世界系左乘：q = qz ⊗ GRASP_Q
        down_q = [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                  a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]
        rec.grasp_yaw = yaw
    rec.tcp_q_init = to_np(u.agent.tcp_pose.q)[0].tolist()
    def plan_to(p, q_now):
        t0 = time.time()
        r = planner.plan_qpos_to_pose(np.concatenate([p, down_q]), q_now, time_step=1 / CONTROL_HZ)
        rec.plans.append(dict(target=[float(v) for v in np.concatenate([p, down_q])],
                              q_start=[float(v) for v in q_now], status=r["status"],
                              n_points=int(len(r.get("position", []))), dt_s=round(time.time() - t0, 3)))
        if r["status"] != "Success":
            raise RuntimeError(f"plan failed: {r['status']}")
        return r
    def log_plan(kind, tgt, q_now, r, t0):
        rec.plans.append(dict(kind=kind, target=[float(v) for v in tgt], q_start=[float(v) for v in q_now],
                              status=r["status"], n_points=int(len(r.get("position", []))),
                              dt_s=round(time.time() - t0, 3)))
    def screw_to(p, q_now):
        """直线笛卡尔路径（抬升用，避免关节空间插值导致夹爪横向甩动）；失败退回关节空间规划。"""
        tgt = np.concatenate([p, down_q]); t0 = time.time()
        r = planner.plan_screw(tgt, q_now, time_step=1 / CONTROL_HZ)
        log_plan("screw", tgt, q_now, r, t0)
        return r if r["status"] == "Success" else plan_to(p, q_now)
    half = float(getattr(u, "cube_half_size", 0.02))
    q8 = lambda q6: np.concatenate([q6, qpos[6:]])
    # 1) 预抓取：方块上方 0.10 m
    pre = plan_to(obj_p + [0, 0, 0.10], qpos); follow(env, pre, GRIP_OPEN, rec)
    q = pre["position"][-1]
    # 2) 下降：end_link 原点即指尖平面。旧版目标 z=中心+0.01，指尖只包住方块顶部约 1 cm → 抬升滑脱。
    #    现在指尖压到方块中心以下 GRASP_DEPTH_FRAC*half，包住约 3/4 高度；离地留 GRASP_MIN_Z。
    z_tip = max(obj_p[2] - GRASP_DEPTH_FRAC * half, GRASP_MIN_Z)
    grasp = plan_to(np.array([obj_p[0], obj_p[1], z_tip]), q8(q)); follow(env, grasp, GRIP_OPEN, rec)
    q = grasp["position"][-1]
    gripper_to(env, q[:6], GRIP_OPEN, SETTLE_STEPS, rec)   # 等 PD 跟上（OMEN 上 joint6 跟踪误差到 0.075 rad）
    # 3) 合爪：步数 10 → CLOSE_STEPS，再保持几步让接触力稳定
    gripper_to(env, q[:6], GRIP_CLOSE, CLOSE_STEPS, rec)
    # 4) 抬升：竖直直线
    lift = screw_to(np.array([obj_p[0], obj_p[1], obj_p[2] + LIFT_DZ]), q8(q)); follow(env, lift, GRIP_CLOSE, rec)
    # 5) 抬起后保持，供 hold_count / k 标定
    gripper_to(env, lift["position"][-1][:6], GRIP_CLOSE, HOLD_STEPS, rec)

def dump_diag(out, tag, rec, extra=None):
    """无论成功失败都落盘：规划日志、跟踪误差、每步 evaluate、首末帧图像。出问题时靠它复盘。"""
    d = dict(keys=np.array(DIAG_KEYS), values=np.array(rec.diag, dtype=np.float32),
             track=np.array(rec.track, dtype=np.float32), tcp_q_init=np.array(getattr(rec, "tcp_q_init", [])))
    if rec.frames:
        for k in ("wrist", "overhead"):
            d[f"{k}_first"] = rec.frames[0][k]; d[f"{k}_last"] = rec.frames[-1][k]
    np.savez_compressed(out / f"diag_{tag}.npz", **d)
    (out / f"plans_{tag}.json").write_text(json.dumps(dict(plans=rec.plans, grasp_yaw=getattr(rec, "grasp_yaw", None), **(extra or {})), ensure_ascii=False, indent=1))

def run_manifest(args):
    import platform, subprocess
    m = dict(args=vars(args), python=platform.python_version(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for mod in ("numpy", "mani_skill", "sapien", "mplib", "torch"):
        try:
            m[mod] = __import__(mod).__version__
        except Exception as e:
            m[mod] = f"n/a ({type(e).__name__})"
    try:
        import torch; m["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    try:
        m["git_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        m["git_commit"] = "n/a"
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/mp_demos_smoke")
    ap.add_argument("--env-id", default="PickCubeReBot-v1")
    ap.add_argument("--cam-rand", action="store_true", help="相机位姿随机化；范围未定前会拒绝")
    ap.add_argument("--no-align-yaw", action="store_true", help="不按方块 yaw 对齐手腕（复现旧数据 mp_demos_n100）")
    args = ap.parse_args()
    global ALIGN_YAW
    ALIGN_YAW = not args.no_align_yaw
    if args.seed + args.n > 10000:
        sys.exit("[gen] seed 段与 B0 测试 seed (>=10000) 重叠，拒绝生成")
    if args.cam_rand:
        load_cam_rand()  # 范围定了以后由 env 侧按此范围施加
    import gymnasium as gym
    import mani_skill.envs  # noqa
    sys.path.insert(0, str(ROOT))
    import sim.maniskill_rebot  # noqa 注册 agent
    import pick_cube_rebot  # noqa 注册 env
    env = gym.make(args.env_id, obs_mode="rgb+state", control_mode="pd_joint_pos",
                   sim_backend="auto", num_envs=1)
    planner = make_planner()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps(run_manifest(args), ensure_ascii=False, indent=2))
    stats = dict(ok=0, fail_plan=0, fail_success=0)
    t_start = time.time()
    def progress(ep, tag, n_steps=None):
        el = time.time() - t_start; eta = el / (ep + 1) * (args.n - ep - 1)
        st = f" {n_steps}步" if n_steps else ""
        print(f"[{ep + 1}/{args.n}] seed={args.seed + ep} {tag}{st} | ok={stats['ok']} fail_plan={stats['fail_plan']} fail_success={stats['fail_success']} | 已用 {el:.0f}s 预计剩余 {eta:.0f}s", flush=True)
    print(f"[gen] 开始采集 {args.n} 条，seed {args.seed}-{args.seed + args.n - 1}，输出到 {out}", flush=True)
    for ep in range(args.n):
        env.reset(seed=args.seed + ep)
        rec = Recorder()
        try:
            solve(env, planner, rec)
        except Exception as e:  # 规划失败或运行时异常都记下，不中断整批
            stats["fail_plan"] += 1; print(f"[gen] ep{ep} {e}")
            dump_diag(out, f"fail_plan_{ep:05d}", rec, dict(error=repr(e), seed=args.seed + ep)); progress(ep, "fail_plan"); continue
        n_steps = len(rec.frames)
        max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        if max_steps and n_steps > max_steps:
            print(f"[gen] ep{ep} 警告：示范 {n_steps} 步 > max_episode_steps={max_steps}，评测/残差会被提前截断")
        succ = bool(to_np(env.unwrapped.evaluate()["success"])[0])
        if not succ:
            stats["fail_success"] += 1
            dump_diag(out, f"fail_success_{ep:05d}", rec, dict(seed=args.seed + ep))
            progress(ep, "fail_success", n_steps); continue
        f = rec.frames
        np.savez_compressed(out / f"episode_{ep:05d}.npz",
                            action=np.stack([x["action"] for x in f]),
                            observation_state=np.stack([x["state"] for x in f]),
                            image_wrist=np.stack([x["wrist"] for x in f]),
                            image_overhead=np.stack([x["overhead"] for x in f]),
                            success=np.array(True), seed=np.array(args.seed + ep))
        dump_diag(out, f"{ep:05d}", rec, dict(seed=args.seed + ep))
        stats["ok"] += 1
        progress(ep, "ok", n_steps)
    meta = dict(env_id=args.env_id, fps=CONTROL_HZ, control_mode="pd_joint_pos",
                action_unit="rad(joint1-6)+m(finger_left)", success_source="env.evaluate()['success']",
                cam_rand=args.cam_rand, align_yaw=ALIGN_YAW, stats_id="pending_single_dataset_rebot_b601",
                lerobot_keys=dict(state="observation.state", action="action",
                                  wrist="observation.images.wrist", overhead="observation.images.overhead"),
                counts=stats, seed0=args.seed, n=args.n, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(stats))

if __name__ == "__main__":
    main()
