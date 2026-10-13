#!/usr/bin/env python3
"""Attempt real LIBERO OffScreenRenderEnv smoke (optional; heavy deps).

Install gap documented here — do NOT run by default in gate CI.
Full `pip install libero` pulls torch/CUDA/robosuite/wandb/transformers.

Usage (after deciding to install):
  cd /workspace/openarm_smolvla_residual
  source .venv/bin/activate
  export MUJOCO_GL=egl
  # Optional heavy install (minutes+, large):
  #   pip install 'libero==0.1.1'   # or git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
  #   # may also need: robosuite==1.4.0, bddl, mujoco matching robosuite
  python scripts/attempt_libero_real_env.py

On ImportError / failure: print exact gap and exit 2 (stub gate remains source of truth).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_ROOT = Path(__file__).resolve().parents[1]
_LOG = _ROOT / "logs" / "libero_gate"
_CONTRACT = _ROOT / "openarm_residual_gate" / "contract_libero_spatial_v0.json"


def main() -> int:
    _LOG.mkdir(parents=True, exist_ok=True)
    report: dict = {"attempt": "real_libero_OffScreenRenderEnv", "ok": False}

    try:
        import libero  # noqa: F401
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as e:
        report["ok"] = False
        report["stage"] = "import"
        report["error"] = f"{type(e).__name__}: {e}"
        report["install_gap"] = (
            "libero/robosuite not importable on this box. "
            "pip package libero==0.1.1 dry-run pulls torch+CUDA toolkit+robosuite==1.4.0+"
            "wandb+transformers+bddl (hundreds of MB–GB). Python here is 3.13; "
            "upstream LIBERO historically targets older CPython — may need a separate venv. "
            "Gate smoke uses openarm_residual_gate/residual_env_libero.py backend=stub."
        )
        report["traceback"] = traceback.format_exc()
        (_LOG / "real_attempt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2

    try:
        contract = json.loads(_CONTRACT.read_text())
        suite_name = contract.get("task_suite", "libero_spatial")
        task_id = int(contract.get("task_id", 0))
        control_hz = int(contract.get("control_hz", 20))
        bench = benchmark.get_benchmark_dict()
        suite = bench[suite_name]()
        task = suite.get_task(task_id)
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        report["task_name"] = task.name
        report["bddl"] = bddl
        report["control_hz"] = control_hz

        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=128,
            camera_widths=128,
            control_freq=control_hz,
        )
        env.seed(0)
        obs = env.reset()
        keys = sorted(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
        report["obs_keys"] = keys
        dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        for i in range(3):
            obs, reward, done, info = env.step(dummy)
            ok = bool(env.check_success())
            report.setdefault("steps", []).append(
                {"i": i, "reward": float(reward), "done": bool(done), "check_success": ok}
            )
        report["action_dim"] = 7
        report["ok"] = True
        env.close()
    except Exception as e:
        report["ok"] = False
        report["stage"] = "env_smoke"
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        (_LOG / "real_attempt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2

    (_LOG / "real_attempt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("REAL LIBERO smoke OK — re-run gate with LIBERO_BACKEND=real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
