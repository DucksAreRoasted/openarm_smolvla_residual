#!/usr/bin/env python3
"""把运动规划示范 (episode_*.npz) 转成 LeRobot 数据集，并计算正式 stats_id。
用法:
  python scripts/转换LeRobot数据集_rebot.py --src data/mp_demos_n100 --out data/lerobot/rebot_b601_pickcube_mp_n100 \
      --repo-id local/rebot_b601_pickcube_mp_n100 --std-floor 1e-3
  只算统计量不转换: 加 --stats-only
兼容 lerobot 0.3.x / 2.x 的 LeRobotDataset.create / add_frame / save_episode 接口。
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "finger_left"]
TASK = "pick up the cube and lift it"

def calc_stats(files, std_floor):
    A, S = [], []
    for f in files:
        e = np.load(f); A.append(e["action"]); S.append(e["observation_state"])
    A, S = np.concatenate(A), np.concatenate(S)
    st = {}
    for k, v in (("action", A), ("observation.state", S)):
        std = v.std(0)
        st[k] = {"mean": v.mean(0).tolist(), "std_raw": std.tolist(),
                 "std": np.maximum(std, std_floor).tolist(),
                 "min": v.min(0).tolist(), "max": v.max(0).tolist(),
                 "q01": np.quantile(v, 0.01, 0).tolist(), "q99": np.quantile(v, 0.99, 0).tolist(),
                 "std_floored_dims": [JOINTS[i] for i in np.where(std < std_floor)[0]]}
    st.update(n_frames=int(len(A)), n_episodes=len(files), std_floor=std_floor, names=JOINTS)
    body = json.dumps(st, sort_keys=True).encode()
    st["stats_id"] = "rebot_b601_mp_n%d_%s" % (len(files), hashlib.sha256(body).hexdigest()[:16])
    return st

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/rebot_b601_pickcube_mp")
    ap.add_argument("--std-floor", type=float, default=1e-3)
    ap.add_argument("--no-video", action="store_true", help="图像存 png 而不是 mp4")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "episode_*.npz")))
    assert files, "没有找到 episode_*.npz"
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    fps = int(meta["fps"])
    st = calc_stats(files, a.std_floor)
    st["source"] = {"src": a.src, "seed0": meta.get("seed0"), "cam_rand": meta.get("cam_rand")}
    print("stats_id =", st["stats_id"], "| 帧数", st["n_frames"], "| std 下限生效维度", st["action"]["std_floored_dims"])
    os.makedirs(a.out, exist_ok=True) if a.stats_only else None
    if a.stats_only:
        json.dump(st, open(os.path.join(a.out, "统计量_stats.json"), "w"), indent=1, ensure_ascii=False); return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa
    e0 = np.load(files[0]); H, W = e0["image_wrist"].shape[1:3]
    img = "image" if a.no_video else "video"
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINTS},
        "observation.images.wrist": {"dtype": img, "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        "observation.images.overhead": {"dtype": img, "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    }
    if os.path.exists(a.out): sys.exit("输出目录已存在，请换路径或手动删除: " + a.out)
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=fps, features=features, root=a.out,
                               robot_type="rebot_b601_dm", use_videos=not a.no_video)
    t0 = time.time()
    for i, f in enumerate(files):
        e = np.load(f)
        for t in range(len(e["action"])):
            fr = {"observation.state": e["observation_state"][t].astype(np.float32),
                  "action": e["action"][t].astype(np.float32),
                  "observation.images.wrist": e["image_wrist"][t],
                  "observation.images.overhead": e["image_overhead"][t]}
            try: ds.add_frame({**fr, "task": TASK})
            except (TypeError, KeyError, ValueError): ds.add_frame(fr)
        try: ds.save_episode()
        except TypeError: ds.save_episode(task=TASK)
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)} {len(e['action'])} 帧  已用 {time.time()-t0:.0f}s", flush=True)
    if hasattr(ds, "finalize"): ds.finalize()
    # LeRobot 训练读的是 meta/stats.json，不读本脚本的统计量；把同样的 std 下限写进去，防止 joint5 被放大上千倍
    sp = os.path.join(a.out, "meta", "stats.json")
    if os.path.exists(sp):
        ls = json.load(open(sp))
        for k in ("action", "observation.state"):
            if k in ls and "std" in ls[k]:
                ls[k]["std"] = np.maximum(np.array(ls[k]["std"], dtype=float), a.std_floor).tolist()
        json.dump(ls, open(sp, "w"), indent=1)
        print("已对 meta/stats.json 的 action / observation.state 施加 std 下限", a.std_floor)
    else:
        print("警告: 没找到 meta/stats.json，std 下限未写入 LeRobot 统计量，请人工检查")
    json.dump(st, open(os.path.join(a.out, "统计量_stats.json"), "w"), indent=1, ensure_ascii=False)
    print("完成:", a.out, "stats_id =", st["stats_id"])

if __name__ == "__main__":
    main()
