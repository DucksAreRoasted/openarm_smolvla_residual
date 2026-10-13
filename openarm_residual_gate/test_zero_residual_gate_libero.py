"""Zero-residual gate on LIBERO Spatial ResidualEnv (Path A draft).

Independent contract_libero_spatial_v0.json — ReBot contract untouched.
Defaults to faithful stub (backend=stub). Real OffScreenRenderEnv via
LIBERO_BACKEND=real after install (see scripts/attempt_libero_real_env.py).

Run (from repo root, with venv):
  cd /workspace/openarm_smolvla_residual
  source .venv/bin/activate
  python openarm_residual_gate/test_zero_residual_gate_libero.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from obs_keys_hash import compute_obs_keys_hash, load_contract
from residual_env_libero import LiberoResidualEnv

_CONTRACT = Path(__file__).with_name("contract_libero_spatial_v0.json")
_REBOT_CONTRACT = Path(__file__).with_name("contract_rebot_b601_dm.json")
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "libero_gate"
_REBOT_HASH_LOCK = "b8b5eef23fc3dd73"


def main() -> int:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    # --- freeze check: ReBot hash must stay locked ---
    rebot = load_contract(_REBOT_CONTRACT)
    rebot_h = compute_obs_keys_hash(rebot)
    if rebot_h != _REBOT_HASH_LOCK:
        log(f"FAIL ReBot obs_keys_hash drifted: {rebot_h} != {_REBOT_HASH_LOCK}")
        return 1
    log(f"OK ReBot hash frozen: {rebot_h}")

    contract = load_contract(_CONTRACT)
    assert contract["robot"] == "libero_franka_panda", contract["robot"]
    assert int(contract["action_dim"]) == 7
    assert int(contract["proprio_dim"]) == 8
    assert int(contract["obs_dim"]) == 16
    assert int(contract["control_hz"]) == 20
    assert contract["reward_version"] == "libero_v0_sparse_success_plus_delta_l2"
    assert contract["stats_id"] == "pending_libero"
    assert contract["stats_id"] != "pending_rebot_b601_dm"
    assert contract.get("success_source") == "env.check_success"
    assert "z_lift" not in contract
    assert "z_lift_abs" not in contract
    # VLA key map lock
    okm = contract["observation_key_map"]
    assert okm["agentview"]["libero_native"] == "agentview_image"
    assert okm["agentview"]["lerobot"] == "observation.images.image"
    assert okm["wrist"]["libero_native"] == "robot0_eye_in_hand_image"
    assert okm["wrist"]["lerobot"] == "observation.images.image2"
    assert okm["proprio"]["lerobot"] == "observation.state"
    assert int(okm["proprio"]["dim"]) == 8
    assert okm["action"]["lerobot"] == "action"
    assert int(okm["action"]["dim"]) == 7

    backend = os.environ.get("LIBERO_BACKEND", contract.get("backend", "stub"))
    atol = float(contract["atol_zero_residual"])
    try:
        env = LiberoResidualEnv(
            clip_c=float(contract["clip_c_default"]),
            chunk_size=int(contract["chunk_size"]),
            control_hz=int(contract["control_hz"]),
            contract_path=_CONTRACT,
            backend=backend,
            force_success_at=5,  # stub reward path: success_just_now at t=5
        )
    except ImportError as e:
        log(f"STUB_FALLBACK real backend unavailable: {e}")
        env = LiberoResidualEnv(
            clip_c=float(contract["clip_c_default"]),
            chunk_size=int(contract["chunk_size"]),
            control_hz=int(contract["control_hz"]),
            contract_path=_CONTRACT,
            backend="stub",
            force_success_at=5,
        )
        backend = "stub"

    log(f"backend={env.backend} task_suite={env.task_suite} task_id={env.task_id}")
    log(f"action_dim={env.n} proprio_dim={env.proprio_dim} obs_dim={env.obs_dim}")
    log(f"clip_c={env.clip_c} unit={env.clip_c_unit} control_hz={env.control_hz} n_substeps={env.n_substeps}")
    log(f"obs_keys_hash={env.obs_keys_hash}")
    log(f"reward_version={env.reward_version} stats_id={env.mean_std_id}")

    # dims
    obs, info = env.reset(seed=0)
    if obs.shape != (16,):
        log(f"FAIL obs shape {obs.shape} != (16,)")
        return 1
    if info["obs_keys_hash"] != env.obs_keys_hash:
        log("FAIL info obs_keys_hash mismatch")
        return 1

    steps = int(contract["chunk_size"]) * 2 + 5

    # δ=0 ≡ base reproducibility
    seq_zero = env.roll_a_exec(steps, force_delta_zero=True, seed=0)
    seq_base = env.roll_a_exec(steps, force_delta_zero=True, seed=0)
    if not np.allclose(seq_zero, seq_base, atol=atol):
        diff = float(np.max(np.abs(seq_zero - seq_base)))
        log(f"FAIL δ=0 roll not reproducible max|Δ|={diff}")
        return 1

    # δ=0 → a_exec == clip(a_base)
    env.reset(seed=0)
    for t in range(min(10, steps)):
        a_base_before = env.a_base.copy()
        _, _, _, _, info = env.step(np.zeros(env.n), force_delta_zero=True)
        expected = np.clip(a_base_before, env.ctrl_lo, env.ctrl_hi)
        if not np.allclose(info["a_exec"], expected, atol=atol):
            log(f"FAIL δ=0 a_exec != clip(a_base) at t={t}")
            log(f"  a_exec {info['a_exec']}")
            log(f"  expect {expected}")
            return 1

    # nonzero δ changes a_exec
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(np.zeros(env.n), force_delta_zero=True)
    env.reset(seed=0)
    action = np.ones(env.n, dtype=np.float32)
    _, rew1, _, _, info1 = env.step(action, force_delta_zero=False)
    if np.allclose(info0["a_exec"], info1["a_exec"], atol=atol):
        log("FAIL: nonzero δ did not change a_exec")
        return 1
    # δ l2 penalty present when no success yet
    if rew1 >= 0:
        log(f"FAIL expected negative δ penalty before success, got {rew1}")
        return 1

    # log fields
    required = set(contract["log_fields_min"])
    missing = required - set(info1.keys())
    if missing:
        log(f"FAIL missing log fields: {sorted(missing)}")
        return 1

    # check_success path + reward term + terminated
    env2 = LiberoResidualEnv(
        clip_c=float(contract["clip_c_default"]),
        chunk_size=int(contract["chunk_size"]),
        control_hz=int(contract["control_hz"]),
        contract_path=_CONTRACT,
        backend="stub",
        force_success_at=3,
    )
    env2.reset(seed=1)
    got_term = False
    for t in range(1, 6):
        _, rew, terminated, truncated, info = env2.step(np.zeros(env2.n), force_delta_zero=True)
        if t == 3:
            if not info["is_success"] or not info["success"] or not terminated:
                log(f"FAIL check_success path at t=3: info={ {k: info[k] for k in ('success','is_success','ever_success','failure_mode')} } terminated={terminated}")
                return 1
            if abs(rew - float(contract["R_term"])) > 1e-6:
                # δ=0 so reward should be exactly R_term on success_just_now
                log(f"FAIL success reward expected {contract['R_term']} got {rew}")
                return 1
            got_term = True
            break
    if not got_term:
        log("FAIL never hit success_just_now")
        return 1

    # timeout → truncated + failure_mode
    env3 = LiberoResidualEnv(
        clip_c=float(contract["clip_c_default"]),
        chunk_size=int(contract["chunk_size"]),
        control_hz=int(contract["control_hz"]),
        contract_path=_CONTRACT,
        backend="stub",
        force_success_at=None,
    )
    env3.t_max = 5
    env3.reset(seed=2)
    truncated_hit = False
    for _ in range(10):
        _, _, terminated, truncated, info = env3.step(np.zeros(env3.n), force_delta_zero=True)
        if truncated:
            if info["failure_mode"] != "timeout":
                log(f"FAIL timeout truncated but failure_mode={info['failure_mode']}")
                return 1
            if terminated:
                log("FAIL timeout should truncate not terminate")
                return 1
            truncated_hit = True
            break
    if not truncated_hit:
        log("FAIL timeout path not hit")
        return 1

    # camera_in_actor_obs must stay false; Actor obs has no image bytes
    assert contract.get("camera_in_actor_obs") is False
    obs4, _ = env.reset(seed=0)
    if obs4.size != 16:
        log(f"FAIL Actor obs must be 16 floats, got {obs4.size}")
        return 1

    status = "PASS" if env.backend == "stub" else "PASS_REAL"
    # Clarify stub vs real in banner
    if env.backend == "stub":
        banner = "PASS zero-residual gate (libero_spatial STUB draft)"
    else:
        banner = "PASS zero-residual gate (libero_spatial REAL)"
    log(banner)
    log(f"  obs_keys_hash={env.obs_keys_hash}")
    log(f"  action_dim={env.n} proprio_dim={env.proprio_dim} obs_dim={env.obs_dim}")
    log(f"  clip_c={env.clip_c} ({env.clip_c_unit}) control_hz={env.control_hz}")
    log(f"  task_id={env.task_id} reward_version={env.reward_version}")
    log(f"  backend={env.backend} status={status}")
    log(f"  ReBot hash untouched={rebot_h}")

    meta = {
        "result": status,
        "backend": env.backend,
        "obs_keys_hash": env.obs_keys_hash,
        "action_dim": env.n,
        "proprio_dim": env.proprio_dim,
        "obs_dim": env.obs_dim,
        "clip_c": env.clip_c,
        "clip_c_unit": env.clip_c_unit,
        "control_hz": env.control_hz,
        "n_substeps": env.n_substeps,
        "task_suite": env.task_suite,
        "task_id": env.task_id,
        "reward_version": env.reward_version,
        "stats_id": env.mean_std_id,
        "success_source": "env.check_success",
        "rebot_obs_keys_hash": rebot_h,
        "observation_key_map": contract["observation_key_map"],
        "note": "draft pending full libero install" if env.backend == "stub" else "real libero",
    }
    (_LOG_DIR / "gate_stdout.txt").write_text("\n".join(lines) + "\n")
    (_LOG_DIR / "gate_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (_LOG_DIR / "obs_keys_hash.txt").write_text(env.obs_keys_hash + "\n")
    env.close()
    return 0


if __name__ == "__main__":
    # Allow running as script from repo root or from package dir
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
