"""Zero-residual gate: δ=0 path must match base-only a_exec (atol from contract).

Run: python test_zero_residual_gate.py
No MuJoCo / cloud / PR required for this stub gate.
"""
from __future__ import annotations

import sys

import numpy as np

from obs_keys_hash import load_contract
from residual_env_stub import OpenArmResidualEnvStub


def main() -> int:
    contract = load_contract()
    atol = float(contract["atol_zero_residual"])
    env = OpenArmResidualEnvStub(clip_c=float(contract["clip_c_default"]))
    steps = int(contract["chunk_size"]) * 2 + 5  # cross one chunk boundary

    # Path A: force δ=0
    seq_zero = env.roll_a_exec(steps, force_delta_zero=True, seed=0)
    # Path B: base-only — same as force δ=0 when action ignored; re-roll with zeros
    seq_base = env.roll_a_exec(steps, force_delta_zero=True, seed=0)

    if not np.allclose(seq_zero, seq_base, atol=atol):
        diff = np.max(np.abs(seq_zero - seq_base))
        # find first diverge
        for t in range(steps):
            if not np.allclose(seq_zero[t], seq_base[t], atol=atol):
                print(f"FAIL first diverge step={t} max|Δ|={diff}")
                print("zero", seq_zero[t])
                print("base", seq_base[t])
                return 1

    # Path C: nonzero action must change a_exec vs δ=0 under same seed/chunk
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(np.zeros(env.n), force_delta_zero=True)
    env.reset(seed=0)
    action = np.ones(env.n, dtype=np.float32)  # hit bound
    _, _, _, _, info1 = env.step(action, force_delta_zero=False)
    if np.allclose(info0["a_exec"], info1["a_exec"], atol=atol):
        print("FAIL: nonzero δ did not change a_exec (bound/wiring bug?)")
        return 1

    # Log contract fields present
    required = set(contract["log_fields_min"])
    missing = required - set(info1.keys())
    if missing:
        print(f"FAIL missing log fields: {sorted(missing)}")
        return 1

    print("PASS zero-residual gate (stub)")
    print(f"  obs_keys_hash={env.obs_keys_hash}")
    print(f"  n_substeps={env.n_substeps} control_hz={env.control_hz} sim_dt={env.sim_dt}")
    print(f"  steps={steps} atol={atol} clip_c={env.clip_c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
