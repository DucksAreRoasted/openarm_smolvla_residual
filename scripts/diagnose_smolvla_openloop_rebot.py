#!/usr/bin/env python3
"""Measure physical-unit action MSE on three training demonstrations.

Each non-overlapping chunk uses its ground-truth initial observation. This is
an open-loop diagnostic, not a closed-loop success evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to("cuda").eval()
    policy.requires_grad_(False)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    chunk_size = policy.config.chunk_size
    rows, squared_errors = [], []
    for source in sorted(args.src.glob("episode_*.npz"))[:3]:
        with np.load(source) as episode:
            seed = int(episode["seed"])
            assert 0 <= seed < 10000
            target = episode["action"]
            state = episode["observation_state"]
            images = {cam: episode[f"image_{cam}"] for cam in ("wrist", "overhead")}
        policy.reset()
        predictions = []
        for start in range(0, len(target), chunk_size):
            batch = {"observation.state": torch.from_numpy(state[start]).unsqueeze(0).cuda(),
                     "task": ["pick up the cube and lift it"]}
            for cam, key in (("wrist", "camera1"), ("overhead", "camera2")):
                batch[f"observation.images.{key}"] = (
                    torch.from_numpy(images[cam][start]).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
                )
            with torch.inference_mode():
                prediction = post(policy.predict_action_chunk(pre(batch)))[0].cpu().numpy()
            predictions.append(prediction[:min(chunk_size, len(target) - start)])
        prediction = np.concatenate(predictions)
        assert prediction.shape == target.shape and np.isfinite(prediction).all()
        error = (prediction.astype(np.float64) - target) ** 2
        squared_errors.append(error)
        trajectory = args.out.with_name(f"openloop_seed_{seed}.npz")
        np.savez_compressed(trajectory, prediction=prediction, target=target)
        rows.append({"seed": seed, "frames": len(target), "mse_per_dimension": error.mean(0).tolist(),
                     "prediction_min": prediction.min(0).tolist(), "prediction_max": prediction.max(0).tolist(),
                     "trajectory": str(trajectory)})
    report = {"checkpoint": args.ckpt, "inference_seed": 0, "chunk_size": chunk_size,
              "protocol": "ground-truth observation at each non-overlapping chunk start",
              "units": ["rad"] * 6 + ["m"], "episodes": rows,
              "mse_per_dimension": np.concatenate(squared_errors).mean(0).tolist()}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
