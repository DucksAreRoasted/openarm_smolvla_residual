"""Thin ResidualEnv stub for interface / zero-residual smoke (no MuJoCo required).

When MuJoCo + v1 XML are available, replace `_physics_step` with real ctrl write + mj_step.
Cameras are intentionally absent — not a visual M1 baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fake_base import FakeBase
from obs_keys_hash import compute_obs_keys_hash, load_contract


class OpenArmResidualEnvStub:
    """Gymnasium-shaped stub without gym dependency for early gate."""

    def __init__(
        self,
        base_policy: Any | None = None,
        clip_c: float = 0.03,
        finger_bound_m: float = 0.05,
        chunk_size: int = 50,
        control_hz: int = 20,
        sim_dt: float = 0.001,
        contract_path: str | Path | None = None,
    ):
        self.contract = load_contract(contract_path)
        self.n = int(self.contract["n_dof"])
        self.base = base_policy or FakeBase(n=self.n, chunk_size=chunk_size)
        self.chunk_size = chunk_size
        self.control_hz = control_hz
        self.sim_dt = sim_dt
        self.n_substeps = int(round(1.0 / (control_hz * sim_dt)))
        assert abs(self.n_substeps * sim_dt * control_hz - 1.0) < 1e-9

        self.bound = np.full(self.n, clip_c, dtype=np.float32)
        self.bound[-1] = finger_bound_m
        self.clip_c = clip_c
        self.ctrl_lo = np.array([-1.396263, -1.74533, -1.570796, 0.0, -1.570796, -0.785398, -1.570796, -0.005], dtype=np.float32)
        self.ctrl_hi = np.array([3.490659, 1.74533, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796, 0.044], dtype=np.float32)

        self.q = np.zeros(self.n, dtype=np.float32)
        self.qdot = np.zeros(self.n, dtype=np.float32)
        self.chunk: np.ndarray | None = None
        self.phase = 0
        self.chunk_idx = 0
        self.a_base = np.zeros(self.n, dtype=np.float32)
        self._seed = 0
        self.reward_version = self.contract["reward_version"]
        self.mean_std_id = str(self.contract.get("stats_id", "physical_smoke"))
        self.obs_keys_hash = compute_obs_keys_hash(self.contract, self.mean_std_id)

    def _obs(self) -> np.ndarray:
        phase = np.float32(self.phase / max(self.chunk_size - 1, 1))
        return np.concatenate([self.q, self.qdot, self.a_base, [phase]]).astype(np.float32)

    def _log_fields(self, delta: np.ndarray, a_exec: np.ndarray, chunk_refresh: bool, chunk_stale: bool) -> dict:
        return {
            "phase": int(self.phase),
            "chunk_idx": int(self.chunk_idx),
            "a_base": self.a_base.copy(),
            "delta": delta.copy(),
            "a_exec": a_exec.copy(),
            "seed": int(self._seed),
            "obs_keys_hash": self.obs_keys_hash,
            "reward_version": self.reward_version,
            "clip_c": float(self.clip_c),
            "chunk_refresh": bool(chunk_refresh),
            "chunk_stale": bool(chunk_stale),
            "n_substeps": self.n_substeps,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
        self.base.reset(self._seed)
        # Fake proprio = home; real env: mj_resetDataKeyframe + cube XY from seed
        self.q = np.array([0, 0, 0, 1.570796, 0, 0, 0, 0.044], dtype=np.float32)
        self.qdot = np.zeros(self.n, dtype=np.float32)
        self.chunk = self.base.infer_chunk({"q": self.q})
        assert self.chunk.shape == (self.chunk_size, self.n)
        self.phase = 0
        self.chunk_idx = 0
        self.a_base = self.chunk[0].copy()
        return self._obs(), {"seed": self._seed, "obs_keys_hash": self.obs_keys_hash}

    def _physics_step(self, a_exec: np.ndarray) -> None:
        # Stub: first-order track toward a_exec (placeholder for mj_step * n_substeps)
        alpha = 0.3
        self.qdot = (a_exec - self.q) * alpha
        self.q = self.q + self.qdot

    def step(self, action: np.ndarray, force_delta_zero: bool = False):
        action = np.asarray(action, dtype=np.float32).reshape(self.n)
        if force_delta_zero:
            delta = np.zeros(self.n, dtype=np.float32)
        else:
            delta = np.clip(action, -1.0, 1.0) * self.bound
        a_exec = np.clip(self.a_base + delta, self.ctrl_lo, self.ctrl_hi)
        self._physics_step(a_exec)

        chunk_refresh = False
        chunk_stale = False
        self.phase += 1
        if self.phase >= self.chunk_size:
            # timeout / exhaust → refresh; stale flag if caller kept using old chunk
            self.chunk = self.base.infer_chunk({"q": self.q})
            self.phase = 0
            self.chunk_idx += 1
            chunk_refresh = True
        self.a_base = self.chunk[self.phase].copy()

        info = self._log_fields(delta, a_exec, chunk_refresh, chunk_stale)
        # reward v0 placeholder (sparse success not wired without cube)
        reward = float(-np.sum(delta * delta))
        terminated = False
        return self._obs(), reward, terminated, False, info

    def roll_a_exec(self, steps: int, force_delta_zero: bool, seed: int) -> np.ndarray:
        self.reset(seed=seed)
        seq = []
        for _ in range(steps):
            _, _, _, _, info = self.step(np.zeros(self.n), force_delta_zero=force_delta_zero)
            seq.append(info["a_exec"].copy())
        return np.stack(seq, axis=0)
