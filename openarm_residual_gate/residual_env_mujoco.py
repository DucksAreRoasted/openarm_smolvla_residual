"""OpenArm v1 ResidualEnv with real MuJoCo mj_step (no cube yet).

Loads vendor openarm_mujoco v1 single-arm XML. Control DoF = 8 (actuators);
qpos has 9 entries because openarm_finger_joint2 is equality-coupled to finger1.
Scene cameras (head/wrist) exist for future VLA; Actor obs stays [q,qdot,a_base,phase]
— images are NOT concatenated (camera_in_actor_obs=false).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Headless before importing mujoco (EGL preferred; OSMesa fallback).
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from fake_base import FakeBase
from obs_keys_hash import compute_obs_keys_hash, load_contract

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_XML = _ROOT / "vendor" / "openarm_mujoco" / "v1" / "openarm.xml"
_DEFAULT_SCENE = _ROOT / "vendor" / "openarm_mujoco" / "v1" / "scene_single_arm.xml"


def default_model_path(prefer_scene: bool = True) -> Path:
    if prefer_scene and _DEFAULT_SCENE.is_file():
        return _DEFAULT_SCENE
    return _DEFAULT_XML


class OpenArmResidualEnvMujoco:
    """Gymnasium-shaped residual env; real physics via mj_step * n_substeps."""

    def __init__(
        self,
        base_policy: Any | None = None,
        clip_c: float = 0.03,
        finger_bound_m: float = 0.05,
        chunk_size: int = 50,
        control_hz: int = 20,
        sim_dt: float | None = None,
        contract_path: str | Path | None = None,
        model_path: str | Path | None = None,
        prefer_scene: bool = True,
    ):
        self.contract = load_contract(contract_path)
        self.n = int(self.contract["n_dof"])
        self.base = base_policy or FakeBase(n=self.n, chunk_size=chunk_size)
        self.chunk_size = chunk_size
        self.control_hz = control_hz

        self.model_path = Path(model_path) if model_path else default_model_path(prefer_scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        xml_dt = float(self.model.opt.timestep)
        self.sim_dt = float(sim_dt) if sim_dt is not None else xml_dt
        if abs(self.sim_dt - xml_dt) > 1e-12:
            self.model.opt.timestep = self.sim_dt
        self.n_substeps = int(round(1.0 / (control_hz * self.sim_dt)))
        assert abs(self.n_substeps * self.sim_dt * control_hz - 1.0) < 1e-9

        self._joint_names = list(self.contract["joint_names"])
        self._actuator_names = list(self.contract["actuator_names"])
        assert len(self._joint_names) == self.n
        assert len(self._actuator_names) == self.n
        self._validate_and_map()

        self.bound = np.full(self.n, clip_c, dtype=np.float32)
        self.bound[-1] = finger_bound_m
        self.clip_c = clip_c
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32).copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32).copy()
        # Ensure actuator order matches contract (remap if needed)
        if self._act_perm is not None:
            self.ctrl_lo = self.ctrl_lo[self._act_perm]
            self.ctrl_hi = self.ctrl_hi[self._act_perm]

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
        self._home_key_id = self._resolve_home_key()

    def _validate_and_map(self) -> None:
        """Map contract joint/actuator names → model addresses; build optional perms."""
        if self.model.nu != self.n:
            raise ValueError(f"model.nu={self.model.nu} != contract n_dof={self.n}")

        model_act = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        if model_act == self._actuator_names:
            self._act_perm = None  # identity: data.ctrl[i] ↔ contract[i]
        else:
            # Permutation: contract order → model actuator index
            name_to_i = {n: i for i, n in enumerate(model_act)}
            missing = [n for n in self._actuator_names if n not in name_to_i]
            if missing:
                raise ValueError(f"actuators missing in model: {missing}; have {model_act}")
            self._act_perm = np.array([name_to_i[n] for n in self._actuator_names], dtype=np.int32)

        self._qpos_adr = np.zeros(self.n, dtype=np.int32)
        self._dof_adr = np.zeros(self.n, dtype=np.int32)
        for i, jn in enumerate(self._joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise ValueError(f"joint {jn!r} not in model")
            self._qpos_adr[i] = self.model.jnt_qposadr[jid]
            self._dof_adr[i] = self.model.jnt_dofadr[jid]

        # Finger2 exists in qpos but not in contract (equality-coupled)
        j2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_finger_joint2")
        self._finger2_qposadr = int(self.model.jnt_qposadr[j2]) if j2 >= 0 else None

    def _resolve_home_key(self) -> int:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid < 0:
            if self.model.nkey > 0:
                return 0
            raise ValueError("no home keyframe in model")
        return int(kid)

    def _read_state(self) -> None:
        self.q = self.data.qpos[self._qpos_adr].astype(np.float32)
        self.qdot = self.data.qvel[self._dof_adr].astype(np.float32)

    def _write_ctrl(self, a_exec: np.ndarray) -> None:
        a = np.asarray(a_exec, dtype=np.float64).reshape(self.n)
        if self._act_perm is None:
            self.data.ctrl[:] = a
        else:
            self.data.ctrl[self._act_perm] = a

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
            "model_path": str(self.model_path),
            "mujoco_version": mujoco.__version__,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
        self.base.reset(self._seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        # Keyframe already fills data.ctrl (model order) and qpos (incl. finger2).
        mujoco.mj_forward(self.model, self.data)
        self._read_state()
        self.chunk = self.base.infer_chunk({"q": self.q})
        assert self.chunk.shape == (self.chunk_size, self.n)
        self.phase = 0
        self.chunk_idx = 0
        self.a_base = self.chunk[0].copy()
        return self._obs(), {
            "seed": self._seed,
            "obs_keys_hash": self.obs_keys_hash,
            "model_path": str(self.model_path),
            "qpos_dim": int(self.model.nq),
            "nu": int(self.model.nu),
        }

    def _physics_step(self, a_exec: np.ndarray) -> None:
        self._write_ctrl(a_exec)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._read_state()

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
            self.chunk = self.base.infer_chunk({"q": self.q})
            self.phase = 0
            self.chunk_idx += 1
            chunk_refresh = True
        self.a_base = self.chunk[self.phase].copy()

        info = self._log_fields(delta, a_exec, chunk_refresh, chunk_stale)
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

    def joint_actuator_report(self) -> dict:
        joints = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        acts = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return {
            "model_path": str(self.model_path),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "timestep": float(self.model.opt.timestep),
            "joints": joints,
            "actuators": acts,
            "contract_joints": self._joint_names,
            "contract_actuators": self._actuator_names,
            "act_perm": None if self._act_perm is None else self._act_perm.tolist(),
            "ctrl_lo": self.ctrl_lo.tolist(),
            "ctrl_hi": self.ctrl_hi.tolist(),
        }
