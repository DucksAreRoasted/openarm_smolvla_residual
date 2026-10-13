# LIBERO Residual Gate Draft (Path A)

**Status (2026-10-12):** stub **PASS** — draft pending full libero install.

## Locked dims

| Field | Value | Notes |
|---|---|---|
| action_dim | 7 | relative EE + gripper; LeRobot `action` |
| proprio_dim | 8 | eef_pos+axis_angle+gripper; LeRobot `observation.state` |
| obs_dim | 16 | Actor `[proprio(8), a_base(7), phase(1)]` — **not** 8+8+7+1 |
| clip_c | 0.05 | unit=`normalized_relative_ee` (≠ ReBot 0.03 rad) |
| control_hz | 20 | residual only on control steps |
| obs_keys_hash | `d9807cfaf4a8701c` | includes `proprio_dim` |

## VLA observation key map (do not invent aliases)

| semantic | LIBERO native | LeRobot key |
|---|---|---|
| agentview | `agentview_image` | `observation.images.image` |
| wrist | `robot0_eye_in_hand_image` | `observation.images.image2` |
| proprio | eef_pos+axis_angle+gripper (8) | `observation.state` |
| action | relative EE+gripper (7) | `action` |

## Success / reward

- Success = `env.check_success()` only — **no z_lift**
- `terminated=success`; timeout → `truncated` + `failure_mode=timeout`
- `failure_mode ∈ {none, timeout, unsafe}`
- `r = -λ‖δ‖² + 1{success_just_now}·R_term` with λ=0.01, R_term=+1.0
- `reward_version=libero_v0_sparse_success_plus_delta_l2`

## Install gap (real env)

`pip install libero==0.1.1` dry-run pulls torch+CUDA+robosuite+wandb+transformers (heavy).
Box Python 3.13; upstream LIBERO often older CPython. Prefer stub for smoke.

```bash
python scripts/attempt_libero_real_env.py   # documents gap → logs/libero_gate/real_attempt.json
LIBERO_BACKEND=real python openarm_residual_gate/test_zero_residual_gate_libero.py  # after install
```

## Frozen

ReBot `contract_rebot_b601_dm.json` / hash `b8b5eef23fc3dd73` untouched.
