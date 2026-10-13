"""4090 smoke for PickCubeReBot-v1: (1) env builds (2) wrist+overhead render (3) delta=0 == base exactly.
No training. Writes logs/maniskill_rebot_smoke/<ts>/{report.json, wrist.png, overhead.png}."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, gymnasium as gym
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim/maniskill_rebot/envs"))
import pick_cube_rebot  # noqa: F401 registers env

out = ROOT / "logs/maniskill_rebot_smoke" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
rep = {"checks": {}}
# ---- run manifest (env debugging) ----
import platform, subprocess, importlib
man = {"python": platform.python_version(), "argv": sys.argv, "time": time.strftime("%Y-%m-%d %H:%M:%S %z")}
for m in ("mani_skill", "sapien", "torch", "numpy", "gymnasium", "mplib"):
    try: man[m] = importlib.import_module(m).__version__
    except Exception as e: man[m] = f"N/A ({e.__class__.__name__})"
try: man["gpu"] = torch.cuda.get_device_name(0); man["cuda_arch_list"] = torch.cuda.get_arch_list()
except Exception as e: man["gpu"] = repr(e)
try: man["git_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
except Exception: man["git_commit"] = None
try: man["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
except Exception: man["nvidia_smi"] = None
json.dump(man, open(out / "run_manifest.json", "w"), indent=2)

def make():
    return gym.make("PickCubeReBot-v1", obs_mode="rgb", control_mode="pd_joint_pos",
                    render_mode="rgb_array", sim_backend="auto")

def fake_base(t, q0):  # deterministic FakeBase standing in for frozen SmolVLA
    a = np.zeros(7, dtype=np.float32); a[:6] = q0[:6] + 0.1 * np.sin(0.2 * t); a[6] = 0.025 + 0.02 * np.sin(0.3 * t)
    return a

try:
    env = make(); obs, _ = env.reset(seed=0)
    u = env.unwrapped
    rep["checks"]["env_build"] = True
    rep["active_joints"] = [j.name for j in u.agent.robot.get_active_joints()]
    rep["action_space"] = str(env.action_space)
    tp = u.agent.tcp_pose
    rep["tcp_pose_init"] = {"p": tp.p[0].tolist(), "q_wxyz": tp.q[0].tolist(), "link": "end_link",
                            "approach_axis_local": "+x", "expected_q_wxyz_approx": [0.7071, 0, 0.7071, 0]}
    rep["init_qpos"] = u.agent.robot.get_qpos()[0].tolist()
    cams = obs["sensor_data"]
    for k in ("wrist", "overhead"):
        img = cams[k]["rgb"][0].cpu().numpy()
        from PIL import Image; Image.fromarray(img).save(out / f"{k}.png")
        rep["checks"][f"cam_{k}"] = bool(img.shape[:2] == (480, 640) and img.std() > 1.0)
        rep[f"{k}_shape"], rep[f"{k}_std"] = list(img.shape), float(img.std())
        red = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
        if red.any():
            ys, xs = np.nonzero(red); bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            full = bb[0] > 2 and bb[1] > 2 and bb[2] < img.shape[1] - 3 and bb[3] < img.shape[0] - 3
        else:
            bb, full = None, False
        rep[f"{k}_cube_bbox_xyxy"] = bb
        rep["checks"][f"cube_fully_in_{k}"] = bool(full)
    env.close()
except Exception as e:
    import traceback; rep["error"] = traceback.format_exc(); json.dump(rep, open(out / "report.json", "w"), indent=2); print(json.dumps(rep, indent=2)); sys.exit(1)

def rollout(residual):
    env = make(); env.reset(seed=123); u = env.unwrapped
    q0 = u.agent.robot.get_qpos()[0].cpu().numpy(); lo, hi = env.action_space.low, env.action_space.high
    traj = []
    from collections import defaultdict; D = defaultdict(list)
    qlim = u.agent.robot.get_qlimits()[0].cpu().numpy()
    for t in range(40):
        a = fake_base(t, q0)
        if residual: a = np.clip(a + np.zeros(7, dtype=np.float32), lo, hi)
        else: a = np.clip(a, lo, hi)
        _, _, _, _, info = env.step(a); qp = u.agent.robot.get_qpos()[0].cpu().numpy().copy(); traj.append(qp)
        D["action"].append(a); D["qpos"].append(qp); D["qvel"].append(u.agent.robot.get_qvel()[0].cpu().numpy())
        D["track_err"].append(qp[:7] - a)  # PD tracking error (arm 6 + finger_left)
        lo_j, hi_j = qlim[:, 0], qlim[:, 1]; D["limit_margin"].append(np.minimum(qp - lo_j, hi_j - qp))
        D["cube_pose"].append(u.cube.pose.raw_pose[0].cpu().numpy())
        for k in ("lift_height", "is_grasped", "contact_force_left", "contact_force_right", "hold_count", "success"):
            D[k].append(np.asarray(info[k][0].cpu()) if hasattr(info[k], "cpu") else info[k])
    np.savez(out / f"diag_rollout_{'residual0' if residual else 'base'}.npz", **{k: np.stack(v) for k, v in D.items()})
    env.close(); return np.stack(traj)

try:
    ta, tb = rollout(False), rollout(True)
    # gripper actuation check: no cube interaction expected, toggle finger_left target 0 <-> 0.05
    env = make(); env.reset(seed=7); u = env.unwrapped
    q0 = u.agent.robot.get_qpos()[0].cpu().numpy(); g = []
    for tgt in (0.05, 0.0, 0.05, 0.0):
        a = np.r_[q0[:6], tgt].astype(np.float32)
        for _ in range(15): env.step(a)
        qp = u.agent.robot.get_qpos()[0].cpu().numpy(); g.append({"target": tgt, "finger_left": float(qp[6]), "finger_right": float(qp[7])})
    env.close(); rep["gripper_toggle"] = g
    rep["checks"]["gripper_follows"] = bool(all(abs(x["finger_left"] - x["target"]) < 0.005 and abs(x["finger_right"] + x["target"]) < 0.005 for x in g))
except Exception as e:
    import traceback; rep["error_rollout"] = traceback.format_exc(); rep["checks"]["delta0_equals_base"] = False
    json.dump(rep, open(out / "report.json", "w"), indent=2); print(rep["error_rollout"]); sys.exit(1)
rep["nan_in_traj"] = bool(np.isnan(ta).any() or np.isnan(tb).any()); rep["checks"]["no_nan"] = not rep["nan_in_traj"]
rep["zero_residual_max_abs_diff"] = float(np.abs(ta - tb).max())
rep["checks"]["delta0_equals_base"] = bool(np.array_equal(ta, tb))
rep["placeholders"] = ["cube_half_size", "cube_spawn_center", "lift_height", "wrist_fov", "pd_gains", "rest_qpos"]
rep["overall"] = "PASS" if all(rep["checks"].values()) else "FAIL"
json.dump(rep, open(out / "report.json", "w"), indent=2); print(json.dumps(rep, indent=2), "\nlogs:", out)
sys.exit(0 if rep["overall"] == "PASS" else 1)
