#!/usr/bin/env python3
"""B0 sparse-reward smoke: δ≡0, fixed seed, dump success/ever_lift.

Shares success definition with residual env via contract_rebot_b601_dm.json
(z_lift_abs + success_hold_steps). Do NOT invent a separate success threshold.

Run (from repo root):
  source .venv/bin/activate
  export MUJOCO_GL=egl
  python scripts/rebot_reward_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "openarm_residual_gate"))

from obs_keys_hash import compute_obs_keys_hash, load_contract  # noqa: E402
from residual_env_rebot_mujoco import ReBotResidualEnvMujoco  # noqa: E402

_CONTRACT = _ROOT / "openarm_residual_gate" / "contract_rebot_b601_dm.json"
_OUT_DIR = _ROOT / "logs" / "rebot_reward_smoke"


def main() -> int:
    contract = load_contract(_CONTRACT)
    seed = 0
    env = ReBotResidualEnvMujoco(
        clip_c=float(contract["clip_c_default"]),
        finger_bound_m=float(contract["finger_bound_m"]),
        chunk_size=int(contract["chunk_size"]),
        control_hz=int(contract["control_hz"]),
        contract_path=_CONTRACT,
    )
    t_max = int(contract["t_max"])
    z_lift_abs = float(contract["z_lift_abs"])
    H = int(contract["success_hold_steps"])

    obs, info0 = env.reset(seed=seed)
    rest_cube_z = float(info0["cube_z"])
    assert obs.shape == (int(contract["obs_dim"]),), obs.shape
    # Actor obs must NOT contain cube privilege
    assert "cube" not in "".join(contract["obs_keys"]).lower()

    rows = []
    terminated = truncated = False
    last_info = info0
    ep_return = 0.0
    for t in range(t_max):
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.n, dtype=np.float32), force_delta_zero=True
        )
        ep_return += float(reward)
        last_info = info
        rows.append(
            {
                "t": int(info["t"]),
                "cube_z": float(info["cube_z"]),
                "success": bool(info["success"]),
                "ever_lift": bool(info["ever_lift"]),
                "held_steps": int(info["held_steps"]),
                "failure_mode": str(info["failure_mode"]),
                "reward": float(reward),
            }
        )
        if terminated or truncated:
            break

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj_path = _OUT_DIR / f"smoke_seed{seed}.jsonl"
    with traj_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Meta shares the SAME success hyperparams as contract (VLA + residual one source)
    meta = {
        "seed": seed,
        "delta": "zero",
        "steps_run": len(rows),
        "t_max": t_max,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(last_info.get("success", False)),
        "ever_lift": bool(last_info.get("ever_lift", False)),
        "held_steps_final": int(last_info.get("held_steps", 0)),
        "failure_mode": str(last_info.get("failure_mode", "none")),
        "cube_z_final": float(last_info.get("cube_z", rest_cube_z)),
        "cube_z_rest_measured": rest_cube_z,
        "ep_return": ep_return,
        "reward_version": contract["reward_version"],
        "obs_keys_hash": env.obs_keys_hash,
        "obs_keys_hash_expected": "b8b5eef23fc3dd73",
        "obs_dim": int(obs.shape[0]),
        "obs_keys": list(contract["obs_keys"]),
        # SAME success definition as residual env / future VLA demo labels
        "z_lift_abs": z_lift_abs,
        "success_hold_steps": H,
        "z_lift_rel": float(contract["z_lift_rel"]),
        "cube_rest_z": float(contract["cube_rest_z"]),
        "table_top_z": float(contract["table_top_z"]),
        "cube_half": float(contract["cube_half"]),
        "lambda_delta": float(contract["lambda_delta"]),
        "R_term": float(contract["R_term"]),
        "drop_margin": float(contract["drop_margin"]),
        "cube_body": contract.get("cube_body"),
        "traj_path": str(traj_path.relative_to(_ROOT)),
    }
    meta_path = _OUT_DIR / "smoke_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    hash_ok = env.obs_keys_hash == "b8b5eef23fc3dd73"
    print("B0 rebot reward smoke")
    print(f"  obs_keys_hash={env.obs_keys_hash} stable={hash_ok}")
    print(f"  rest cube_z measured={rest_cube_z:.6f} (contract cube_rest_z={contract['cube_rest_z']})")
    print(f"  z_lift_abs={z_lift_abs} H={H} λ={contract['lambda_delta']} R_term={contract['R_term']}")
    print(f"  steps={len(rows)} success={meta['success']} ever_lift={meta['ever_lift']} "
          f"failure_mode={meta['failure_mode']} return={ep_return:.6f}")
    print(f"  wrote {meta_path}")
    print(f"  wrote {traj_path}")
    if not hash_ok:
        print("WARN: obs_keys_hash changed — investigate before merging")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
