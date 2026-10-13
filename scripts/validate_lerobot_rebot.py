#!/usr/bin/env python3
"""Check the converted dataset against its source and training statistics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    info = json.loads((args.dataset / "meta/info.json").read_text())
    stats = json.loads((args.dataset / "meta/stats.json").read_text())
    provenance = json.loads((args.dataset / "统计量_stats.json").read_text())
    sources = sorted(args.src.glob("episode_*.npz"))
    seeds, actions, states = [], [], []
    for source in sources:
        with np.load(source) as episode:
            seeds.append(int(episode["seed"]))
            assert bool(episode["success"]), source
            actions.append(episode["action"])
            states.append(episode["observation_state"])
    assert len(seeds) == len(set(seeds)) and all(0 <= s < 10000 for s in seeds)
    expected = {"action": np.concatenate(actions), "observation.state": np.concatenate(states)}
    frames = len(expected["action"])
    assert info["total_episodes"] == len(sources)
    assert info["total_frames"] == frames and info["fps"] == 20
    assert provenance["n_frames"] == frames and provenance["n_episodes"] == len(sources)
    for key in ("observation.images.wrist", "observation.images.overhead", "task_index"):
        assert key in info["features"], key
    tables = [pq.read_table(path) for path in sorted((args.dataset / "data").glob("chunk-*/*.parquet"))]
    report = {"stats_id": provenance["stats_id"], "episodes": len(sources), "frames": frames,
              "fps": info["fps"], "seeds": seeds, "checks": {}}
    for key, source in expected.items():
        actual = np.concatenate([np.asarray(table[key].to_pylist(), dtype=np.float32) for table in tables])
        assert actual.shape == (frames, 7) and np.array_equal(actual, source), key
        assert np.isfinite(actual).all(), key
        mean = np.asarray(stats[key]["mean"])
        std = np.asarray(stats[key]["std"])
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        assert np.all(std >= provenance["std_floor"]), key
        normalized = (actual - mean) / (std + 1e-8)
        assert np.isfinite(normalized).all() and np.max(np.abs(normalized)) <= 20, key
        report["checks"][key] = {"source_identical": True, "shape": list(actual.shape),
                                  "std": std.tolist(), "normalized_min": normalized.min(0).tolist(),
                                  "normalized_max": normalized.max(0).tolist()}
    report["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {len(sources)} episodes, {frames} frames, {provenance['stats_id']}")


if __name__ == "__main__":
    main()
