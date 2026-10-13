"""Stable hash for obs_keys / unit / MEAN_STD id / camera layout — bump when contract changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _camera_digest(contract: dict) -> dict | None:
    """Compact camera layout for hashing (names, keys, size, fps, mount). None if no cameras."""
    if not contract.get("camera_in_env", False):
        return None
    cams = contract.get("cameras") or []
    cam_list = []
    for c in cams:
        entry = {
            "name": c["name"],
            "lerobot_key": c.get("lerobot_key"),
            "width": c.get("width"),
            "height": c.get("height"),
            "fps": c.get("fps", contract.get("camera_fps")),
            "fovy": c.get("fovy"),
            "mount": c.get("mount"),
            "body": c.get("body"),
            "pos": c.get("pos"),
            "xyaxes": c.get("xyaxes"),
            "euler": c.get("euler"),
            "extrinsics_source": contract.get("extrinsics_source"),
        }
        # Only include when present so ReBot/OpenArm digests stay byte-identical.
        if "libero_native_key" in c:
            entry["libero_native_key"] = c.get("libero_native_key")
        cam_list.append(entry)
    return {
        "camera_in_env": True,
        "camera_in_actor_obs": bool(contract.get("camera_in_actor_obs", False)),
        "camera_fps": contract.get("camera_fps"),
        "cameras": cam_list,
    }


def compute_obs_keys_hash(contract: dict, mean_std_id: str | None = None) -> str:
    payload = {
        "obs_keys": contract["obs_keys"],
        "n_dof": contract["n_dof"],
        "action_unit": contract["action_unit"],
        "mean_std_id": mean_std_id or contract.get("stats_id", "physical_smoke"),
        "stats_source": contract.get("stats_source", "none"),
        "chunk_size": contract["chunk_size"],
        "control_hz": contract["control_hz"],
        "camera": _camera_digest(contract),
    }
    # Optional: non-q/qdot proprio (LIBERO eef+axis_angle+gripper). Only when present
    # so ReBot/OpenArm (which already have obs_dim but no proprio_dim) keep hashes.
    if "proprio_dim" in contract:
        payload["proprio_dim"] = int(contract["proprio_dim"])
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_contract(path: str | Path | None = None) -> dict:
    path = Path(path or Path(__file__).with_name("contract.json"))
    return json.loads(path.read_text())


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else None
    c = load_contract(p)
    print(compute_obs_keys_hash(c))
