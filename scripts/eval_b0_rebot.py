"""B0 eval: frozen SmolVLA base only (delta=0) on PickCubeReBot-v1. Smoke/short budget only.
Usage: python scripts/eval_b0_rebot.py --policy hold|smolvla [--ckpt <dir>] --n-episodes 5 --seed-start 10000
Test seeds must be disjoint from demo-generation seeds."""
import sys as _s, pathlib as _p
_R=_p.Path(__file__).resolve().parents[1]
_s.path.insert(0,str(_R)); _s.path.insert(0,str(_R/"sim/maniskill_rebot/envs"))
import argparse, json, math, time, pathlib

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["hold", "smolvla"])
    ap.add_argument("--ckpt", default=None, help="smolvla: pretrained_model dir of the fine-tuned checkpoint")
    ap.add_argument("--task", default=None, help="smolvla: language instruction (default = training TASK)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="smolvla: override config.n_action_steps (default: ckpt config)")
    ap.add_argument("--env-id", default="PickCubeReBot-v1")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=None, help="default: env max_episode_steps")
    ap.add_argument("--cam-rand", action="store_true")
    ap.add_argument("--out", default="logs/b0_rebot")
    a = ap.parse_args()
    import gymnasium as gym, mani_skill.envs  # noqa
    from sim.maniskill_rebot import rebot_b601_agent  # noqa: registers agent
    import pick_cube_rebot  # noqa: registers env
    T = a.max_steps
    if T is None:
        try:
            from mani_skill.utils.registration import REGISTERED_ENVS
            T = REGISTERED_ENVS[a.env_id].max_episode_steps
        except Exception:
            T = None
    if T is None:
        raise SystemExit("max_episode_steps unknown: pass --max-steps or register env with it")
    env = gym.make(a.env_id, obs_mode="rgb", control_mode="pd_joint_pos", render_mode=None, max_episode_steps=T)
    print(f"[b0] T={T}")
    if a.policy == "hold":  # pipeline smoke only: hold current qpos, not a real base, never used for judging
        class _Hold:
            def reset(self): pass
            def act(self, obs): return env.unwrapped.agent.robot.get_qpos()[..., :7]
        base = _Hold()
    else:  # smolvla: frozen fine-tuned base, eval(), no grad, chunk queue inside select_action
        if not a.ckpt: raise SystemExit("--policy smolvla needs --ckpt <checkpoint pretrained_model dir>")
        from scripts.smolvla_base_rebot import SmolVLABase, TASK
        base = SmolVLABase(a.ckpt, env, task=a.task or TASK, n_action_steps=a.n_action_steps)
        print(f"[b0] smolvla ckpt={a.ckpt} n_action_steps={base.n_action_steps} task={base.task!r}")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    import platform, subprocess, sys as _sys
    def _ver(m):
        try: return __import__(m).__version__
        except Exception: return None
    try: gpu = __import__("torch").cuda.get_device_name(0)
    except Exception: gpu = None
    try: commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: commit = None
    (out/"run_manifest.json").write_text(json.dumps(dict(argv=_sys.argv, args=vars(a), python=platform.python_version(),
        numpy=_ver("numpy"), torch=_ver("torch"), mani_skill=_ver("mani_skill"), sapien=_ver("sapien"),
        gpu=gpu, git_commit=commit, time=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    import numpy as np
    def _np(x):
        try: return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float32).reshape(-1)
        except Exception: return np.zeros(1, np.float32)
    def _img(o, k):
        try:
            x = o["sensor_data"][k]["rgb"]; x = x.detach().cpu() if hasattr(x, "detach") else x
            return np.asarray(x).astype(np.uint8)
        except Exception: return None
    for i in range(a.n_episodes):
        seed = a.seed_start + i
        obs, info = env.reset(seed=seed); base.reset()
        ever, t0, steps = False, time.time(), 0
        diag = {k: [] for k in ["qpos","action","lift_height","is_grasped","contact_force_left","contact_force_right","hold_count","success","reward"]}
        first = {k: _img(obs, k) for k in ("wrist","overhead")}
        err = None
        for t in range(T):
            try:
                act = base.act(obs)  # delta = 0, a_exec = a_base
                obs, r, term, trunc, info = env.step(act); steps += 1
            except Exception as e:
                err = repr(e); break
            diag["qpos"].append(_np(env.unwrapped.agent.robot.get_qpos())); diag["action"].append(_np(act))
            for k in ["lift_height","is_grasped","contact_force_left","contact_force_right","hold_count","success"]:
                diag[k].append(float(_np(info.get(k, np.nan))[0]))
            diag["reward"].append(float(_np(r)[0]))
            if a.policy == "smolvla":
                diag.setdefault("chunk_idx", []).append(base.chunk_idx); diag.setdefault("phase", []).append(base.phase)
                diag.setdefault("latency_vla_ms", []).append(base.last_latency_ms)
            ever = ever or bool(info.get("success", False))
            if term or trunc: break
        last = {k: _img(obs, k) for k in ("wrist","overhead")}
        tag = "ok" if ever else "fail"
        np.savez_compressed(out/f"diag_{tag}_{seed}.npz", **{k: np.array(v) for k, v in diag.items()},
            **{f"first_{k}": v for k, v in first.items() if v is not None}, **{f"last_{k}": v for k, v in last.items() if v is not None})
        rows.append(dict(seed=seed, error=err, success=bool(info.get("success", False)), ever_success=ever,
                         steps=steps, wall_s=round(time.time()-t0, 2),
                         failure_mode=None if ever else "unlabeled"))
    k = sum(r["success"] for r in rows); n = len(rows)
    lo, hi = wilson(k, n)
    summary = dict(env_id=a.env_id, T=T, policy=a.policy, n=n, successes=k, rate=k/n if n else None,
                   wilson95=[lo, hi], seeds=[r["seed"] for r in rows], cam_rand=a.cam_rand)
    (out/"episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out/"summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
