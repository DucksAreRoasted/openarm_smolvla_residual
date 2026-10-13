"""LIBERO ResidualEnv draft (Path A: pure sim).

Faithful interface stub mirroring OffScreenRenderEnv + residual gate:
  a_exec = clip(a_base + δ), δ=0 ≡ base, success = check_success().

Real LIBERO+robosuite install is heavy (torch/CUDA); this module defaults to
backend=stub. See scripts/attempt_libero_real_env.py for the real-env attempt.
ReBot contract/overlay/gate are frozen — do not modify as primary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fake_base import LIBERO_FAKE_OFFSET, LIBERO_NOOP_ACTION, FakeBase
from obs_keys_hash import compute_obs_keys_hash, load_contract

_DEFAULT_CONTRACT = Path(__file__).with_name("contract_libero_spatial_v0.json")


class LiberoResidualEnv:
    """Gymnasium-shaped residual env for LIBERO Spatial smoke (stub or real).

    Actor obs = [proprio(8), a_base(7), phase(1)] → obs_dim=16.
    Images are NOT in Actor; camera keys deferred to VLA mapping.
    """

    def __init__(
        self,
        base_policy: Any | None = None,
        clip_c: float | None = None,
        chunk_size: int | None = None,
        control_hz: int | None = None,
        contract_path: str | Path | None = None,
        backend: str | None = None,
        force_success_at: int | None = None,
    ):
        self.contract = load_contract(contract_path or _DEFAULT_CONTRACT)
        self.n = int(self.contract["action_dim"])  # residual / a_base dim = 7
        self.proprio_dim = int(self.contract["proprio_dim"])  # 8
        self.obs_dim = int(self.contract["obs_dim"])  # 16
        assert self.obs_dim == self.proprio_dim + self.n + 1
        assert list(self.contract["obs_keys"]) == ["proprio", "a_base", "phase"]

        self.chunk_size = int(chunk_size or self.contract["chunk_size"])
        self.control_hz = int(control_hz or self.contract["control_hz"])
        self.sim_dt = float(self.contract.get("sim_dt", 0.002))
        self.n_substeps = int(self.contract.get("n_substeps") or round(1.0 / (self.control_hz * self.sim_dt)))
        assert abs(self.n_substeps * self.sim_dt * self.control_hz - 1.0) < 1e-6

        self.clip_c = float(clip_c if clip_c is not None else self.contract["clip_c_default"])
        self.clip_c_unit = str(self.contract.get("clip_c_unit", "normalized_relative_ee"))
        self.bound = np.full(self.n, self.clip_c, dtype=np.float32)
        # LIBERO action Box(-1, 1)
        self.ctrl_lo = np.full(self.n, -1.0, dtype=np.float32)
        self.ctrl_hi = np.full(self.n, 1.0, dtype=np.float32)

        if base_policy is not None:
            self.base = base_policy
        else:
            self.base = FakeBase(
                n=self.n,
                chunk_size=self.chunk_size,
                home=LIBERO_NOOP_ACTION.copy(),
                offset=LIBERO_FAKE_OFFSET.copy(),
            )

        self.backend = str(backend or self.contract.get("backend", "stub"))
        self._real_env = None
        if self.backend == "real":
            self._init_real_backend()
        elif self.backend != "stub":
            raise ValueError(f"unknown backend={self.backend!r}; use stub|real")

        self.proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        self.chunk: np.ndarray | None = None
        self.phase = 0
        self.chunk_idx = 0
        self.a_base = np.zeros(self.n, dtype=np.float32)
        self._seed = 0
        self.reward_version = self.contract["reward_version"]
        self.mean_std_id = str(self.contract.get("stats_id", "pending_libero"))
        self.obs_keys_hash = compute_obs_keys_hash(self.contract, self.mean_std_id)
        self.task_id = int(self.contract.get("task_id", 0))
        self.task_suite = str(self.contract.get("task_suite", "libero_spatial"))
        self.lambda_delta = float(self.contract.get("lambda_delta", 0.01))
        self.R_term = float(self.contract.get("R_term", 1.0))
        self.t_max = int(self.contract.get("t_max", 280))
        # Gate-only: force check_success True at this step (stub reward path smoke). None = never.
        self.force_success_at = force_success_at
        self._reset_success_state()

    # --- success / failure (LIBERO check_success; no z_lift) -----------------

    def _reset_success_state(self) -> None:
        self.success = False
        self.is_success = False
        self.ever_success = False
        self.failure_mode = "none"
        self._t = 0

    def check_success(self) -> bool:
        """Authoritative success. Real: delegate to OffScreenRenderEnv.check_success().
        Stub: False unless force_success_at matches current step (gate reward smoke only).
        """
        if self.backend == "real" and self._real_env is not None:
            return bool(self._real_env.check_success())
        if self.force_success_at is not None and self._t == int(self.force_success_at):
            return True
        return False

    def _check_unsafe(self, a_exec: np.ndarray | None = None) -> bool:
        if not np.isfinite(self.proprio).all():
            return True
        if a_exec is not None and not np.isfinite(a_exec).all():
            return True
        if not np.isfinite(self.a_base).all():
            return True
        return False

    def _update_success(self) -> bool:
        """Returns success_just_now. terminated uses env.check_success() only — no z_lift."""
        success_just_now = False
        ok = self.check_success()
        self.is_success = bool(ok)
        if ok and not self.success:
            self.success = True
            self.ever_success = True
            success_just_now = True
        elif ok:
            self.ever_success = True
        if self.failure_mode == "none" and self._check_unsafe():
            self.failure_mode = "unsafe"
        return success_just_now

    # --- obs / log -----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        phase = np.float32(self.phase / max(self.chunk_size - 1, 1))
        return np.concatenate([self.proprio, self.a_base, [phase]]).astype(np.float32)

    def _log_fields(
        self, delta: np.ndarray, a_exec: np.ndarray, chunk_refresh: bool, chunk_stale: bool
    ) -> dict:
        return {
            "phase": int(self.phase),
            "chunk_idx": int(self.chunk_idx),
            "a_base": self.a_base.copy(),
            "delta": delta.copy(),
            "a_exec": a_exec.copy(),
            "seed": int(self._seed),
            "obs_keys_hash": self.obs_keys_hash,
            "chunk_refresh": bool(chunk_refresh),
            "chunk_stale": bool(chunk_stale),
            "success": bool(self.success),
            "is_success": bool(self.is_success),
            "ever_success": bool(self.ever_success),
            "failure_mode": str(self.failure_mode),
            "task_id": int(self.task_id),
            "task_suite": self.task_suite,
            "reward_version": self.reward_version,
            "clip_c": float(self.clip_c),
            "clip_c_unit": self.clip_c_unit,
            "n_substeps": self.n_substeps,
            "control_hz": self.control_hz,
            "backend": self.backend,
            "t": int(self._t),
            "proprio_dim": self.proprio_dim,
            "action_dim": self.n,
            "obs_dim": self.obs_dim,
            # VLA key placeholders (images not fetched in stub Actor path)
            "observation_key_map": self.contract.get("observation_key_map"),
        }

    def _physics_step(self, a_exec: np.ndarray) -> None:
        """Apply a_exec. Stub: drift proprio slightly; real: OffScreenRenderEnv.step."""
        if self.backend == "real" and self._real_env is not None:
            raw_obs, _, _, _ = self._real_env.step(a_exec.tolist())
            self.proprio = self._proprio_from_raw(raw_obs)
            return
        # Stub: proprio stays finite; mild deterministic drift from a_exec EE deltas
        self.proprio[:3] = self.proprio[:3] + 0.001 * a_exec[:3]
        self.proprio[3:6] = self.proprio[3:6] + 0.0005 * a_exec[3:6]
        # gripper channels: map last action dim into both finger slots softly
        self.proprio[6:8] = np.clip(
            self.proprio[6:8] * 0.99 + 0.01 * float(a_exec[6]), -1.0, 1.0
        )

    def _proprio_from_raw(self, raw_obs: dict) -> np.ndarray:
        """Build locked 8-dim observation.state from LIBERO native keys."""
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
        eef_quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32).reshape(4)
        axis_angle = _quat_to_axis_angle(eef_quat)
        grip = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if grip.size == 1:
            grip = np.array([grip[0], grip[0]], dtype=np.float32)
        grip = grip[:2]
        return np.concatenate([eef_pos, axis_angle, grip]).astype(np.float32)

    def _init_real_backend(self) -> None:
        """Lazy import real LIBERO OffScreenRenderEnv. Raises ImportError with install gap."""
        try:
            from libero.libero import benchmark, get_libero_path  # type: ignore
            from libero.libero.envs import OffScreenRenderEnv  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Real LIBERO backend requested but libero/robosuite not importable. "
                "Install gap: pip install libero pulls torch/CUDA/robosuite/wandb/transformers "
                "(heavy). Prefer stub smoke; see scripts/attempt_libero_real_env.py. "
                f"Original: {e}"
            ) from e
        suite_name = self.task_suite
        bench = benchmark.get_benchmark_dict()
        suite = bench[suite_name]()
        task = suite.get_task(self.task_id)
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        self._real_env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            control_freq=self.control_hz,
        )
        self._real_task_name = task.name

    # --- gym-like API --------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
        self.base.reset(self._seed)
        self._reset_success_state()

        if self.backend == "real" and self._real_env is not None:
            self._real_env.seed(self._seed)
            raw = self._real_env.reset()
            self.proprio = self._proprio_from_raw(raw)
        else:
            # Stub home proprio: eef above table-ish, axis_angle 0, gripper open-ish
            rng = np.random.default_rng(self._seed)
            self.proprio = np.zeros(self.proprio_dim, dtype=np.float32)
            self.proprio[:3] = np.array([0.0, 0.0, 0.4], dtype=np.float32) + 0.01 * rng.standard_normal(3).astype(
                np.float32
            )
            self.proprio[6:8] = np.array([0.04, -0.04], dtype=np.float32)

        self.chunk = self.base.infer_chunk({"proprio": self.proprio})
        assert self.chunk.shape == (self.chunk_size, self.n)
        self.phase = 0
        self.chunk_idx = 0
        self.a_base = self.chunk[0].copy()
        obs = self._obs()
        assert obs.shape == (self.obs_dim,)
        return obs, {
            "seed": self._seed,
            "obs_keys_hash": self.obs_keys_hash,
            "success": False,
            "is_success": False,
            "ever_success": False,
            "failure_mode": "none",
            "task_id": self.task_id,
            "reward_version": self.reward_version,
            "backend": self.backend,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.n,
            "obs_dim": self.obs_dim,
            "clip_c": float(self.clip_c),
            "clip_c_unit": self.clip_c_unit,
        }

    def step(self, action: np.ndarray, force_delta_zero: bool = False):
        action = np.asarray(action, dtype=np.float32).reshape(self.n)
        if force_delta_zero:
            delta = np.zeros(self.n, dtype=np.float32)
        else:
            delta = np.clip(action, -1.0, 1.0) * self.bound
        a_exec = np.clip(self.a_base + delta, self.ctrl_lo, self.ctrl_hi)
        self._physics_step(a_exec)
        self._t += 1

        chunk_refresh = False
        chunk_stale = False
        self.phase += 1
        if self.phase >= self.chunk_size:
            self.chunk = self.base.infer_chunk({"proprio": self.proprio})
            self.phase = 0
            self.chunk_idx += 1
            chunk_refresh = True
        self.a_base = self.chunk[self.phase].copy()

        success_just_now = self._update_success()
        # r = -λ‖δ‖² + 1{success_just_now}·R_term  (no extra shaping)
        reward = float(-self.lambda_delta * np.sum(delta * delta))
        if success_just_now:
            reward += self.R_term

        terminated = bool(self.success)
        truncated = False
        if (not terminated) and (self._t >= self.t_max):
            truncated = True
            if self.failure_mode == "none":
                self.failure_mode = "timeout"
        if self.failure_mode == "unsafe":
            terminated = True

        info = self._log_fields(delta, a_exec, chunk_refresh, chunk_stale)
        return self._obs(), reward, terminated, truncated, info

    def roll_a_exec(self, steps: int, force_delta_zero: bool, seed: int) -> np.ndarray:
        self.reset(seed=seed)
        seq = []
        for _ in range(steps):
            _, _, _, _, info = self.step(np.zeros(self.n), force_delta_zero=force_delta_zero)
            seq.append(info["a_exec"].copy())
        return np.stack(seq, axis=0)

    def close(self) -> None:
        if self._real_env is not None:
            try:
                self._real_env.close()
            finally:
                self._real_env = None


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) or (w,x,y,z) to axis-angle (3,).

    LIBERO/robosuite typically stores (x,y,z,w). Stable for smoke / real path.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    # Heuristic: if |w| is last component and largest magnitude often w — robosuite is xyzw
    xyz, w = q[:3], q[3]
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    xyz, w = xyz / norm, w / norm
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / s
    return (axis * angle).astype(np.float32)
