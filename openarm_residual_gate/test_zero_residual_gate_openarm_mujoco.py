"""ARCHIVED OpenArm v1 zero-residual gate (default gate is ReBot).

Zero-residual gate on real OpenArm v1 MuJoCo.

Run (from this directory, with venv + MUJOCO_GL):
  cd /workspace/openarm_smolvla_residual
  source .venv/bin/activate
  export MUJOCO_GL=egl
  python openarm_residual_gate/test_zero_residual_gate_mujoco.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from obs_keys_hash import load_contract
from residual_env_mujoco import OpenArmResidualEnvMujoco


def main() -> int:
    contract = load_contract()
    atol = float(contract["atol_zero_residual"])
    env = OpenArmResidualEnvMujoco(clip_c=float(contract["clip_c_default"]))
    report = env.joint_actuator_report()
    print("model:", report["model_path"])
    print("mujoco joints:", report["joints"])
    print("mujoco actuators:", report["actuators"])
    print("nq/nu/dt:", report["nq"], report["nu"], report["timestep"])
    if report["act_perm"] is not None:
        print("NOTE: actuator permute", report["act_perm"])

    # Naming alignment
    if report["actuators"] != report["contract_actuators"]:
        print("FAIL actuator name mismatch vs contract")
        print("  model   ", report["actuators"])
        print("  contract", report["contract_actuators"])
        return 1
    # Controlled joints (exclude equality-coupled finger2)
    controlled = [j for j in report["joints"] if j != "openarm_finger_joint2"]
    if controlled != report["contract_joints"]:
        print("FAIL controlled joint names mismatch vs contract")
        print("  model   ", controlled)
        print("  contract", report["contract_joints"])
        return 1

    steps = int(contract["chunk_size"]) * 2 + 5

    seq_zero = env.roll_a_exec(steps, force_delta_zero=True, seed=0)
    seq_base = env.roll_a_exec(steps, force_delta_zero=True, seed=0)
    if not np.allclose(seq_zero, seq_base, atol=atol):
        diff = np.max(np.abs(seq_zero - seq_base))
        for t in range(steps):
            if not np.allclose(seq_zero[t], seq_base[t], atol=atol):
                print(f"FAIL first diverge step={t} max|Δ|={diff}")
                print("zero", seq_zero[t])
                print("base", seq_base[t])
                return 1

    # a_exec == a_base when δ=0 (within ctrl clip)
    env.reset(seed=0)
    for t in range(min(10, steps)):
        a_base_before = env.a_base.copy()
        _, _, _, _, info = env.step(np.zeros(env.n), force_delta_zero=True)
        expected = np.clip(a_base_before, env.ctrl_lo, env.ctrl_hi)
        if not np.allclose(info["a_exec"], expected, atol=atol):
            print(f"FAIL δ=0 a_exec != clip(a_base) at t={t}")
            print("a_exec", info["a_exec"])
            print("expect", expected)
            return 1

    # Nonzero residual changes a_exec
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(np.zeros(env.n), force_delta_zero=True)
    env.reset(seed=0)
    action = np.ones(env.n, dtype=np.float32)
    _, _, _, _, info1 = env.step(action, force_delta_zero=False)
    if np.allclose(info0["a_exec"], info1["a_exec"], atol=atol):
        print("FAIL: nonzero δ did not change a_exec")
        return 1

    required = set(contract["log_fields_min"])
    missing = required - set(info1.keys())
    if missing:
        print(f"FAIL missing log fields: {sorted(missing)}")
        return 1

    # Real physics: after several control steps with off-home ctrl, q moves
    env.reset(seed=0)
    q0 = env.q.copy()
    for _ in range(20):
        env.step(np.zeros(env.n), force_delta_zero=True)
    q1 = env.q.copy()
    # FakeBase drifts targets; q should track somewhat (not identical to q0)
    if np.allclose(q1, q0, atol=1e-5):
        # Force a large ctrl offset and step to prove mj_step is live
        env.reset(seed=0)
        q0 = env.q.copy()
        target = q0.copy()
        target[3] = float(np.clip(q0[3] - 0.2, env.ctrl_lo[3], env.ctrl_hi[3]))
        env.a_base = target  # inject
        env._physics_step(np.clip(target, env.ctrl_lo, env.ctrl_hi))
        if np.allclose(env.q, q0, atol=1e-4):
            print("FAIL: mj_step did not move q under off-home ctrl")
            print("q0", q0)
            print("q ", env.q)
            return 1

    print("PASS zero-residual gate (mujoco)")
    print(f"  obs_keys_hash={env.obs_keys_hash}")
    print(f"  n_substeps={env.n_substeps} control_hz={env.control_hz} sim_dt={env.sim_dt}")
    print(f"  steps={steps} atol={atol} clip_c={env.clip_c}")
    print(f"  q0→q1 max|Δ|={float(np.max(np.abs(q1 - q0))):.6f} (FakeBase track)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
