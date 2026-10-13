"""ReBot B601-DM ResidualEnv with real MuJoCo mj_step.

Loads local overlay sim/rebot_b601/scene_residual.xml (actuators + finger equality).
Control DoF = 7 (joint1..6 + finger_left); finger_right equality-coupled.
Actor obs stays [q,qdot,a_base,phase] — images NOT concatenated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from fake_base import FakeBase
from obs_keys_hash import compute_obs_keys_hash, load_contract

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONTRACT = Path(__file__).with_name("contract_rebot_b601_dm.json")


class ReBotResidualEnvMujoco:
    """Gymnasium-shaped residual env for ReBot B601-DM; real physics via mj_step * n_substeps."""

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
    ):
        self.contract = load_contract(contract_path or _DEFAULT_CONTRACT)
        self.n = int(self.contract["n_dof"])
        self.base = base_policy or FakeBase(n=self.n, chunk_size=chunk_size)
        self.chunk_size = chunk_size
        self.control_hz = control_hz

        if model_path:
            self.model_path = Path(model_path)
        else:
            scene = self.contract.get("model_xml_scene")
            if not scene:
                raise ValueError("contract missing model_xml_scene")
            self.model_path = _ROOT / scene
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path.resolve()))
        self.data = mujoco.MjData(self.model)

        xml_dt = float(self.model.opt.timestep)
        self.sim_dt = float(sim_dt) if sim_dt is not None else xml_dt
        if abs(self.sim_dt - xml_dt) > 1e-12:
            self.model.opt.timestep = self.sim_dt
        self.n_substeps = int(round(1.0 / (control_hz * self.sim_dt)))
        assert abs(self.n_substeps * self.sim_dt * control_hz - 1.0) < 1e-9

        self._joint_names = list(self.contract["joint_names"])
        self._actuator_names = list(self.contract["actuator_names"])
        self._equality_coupled = list(self.contract.get("equality_coupled_joints") or [])
        assert len(self._joint_names) == self.n
        assert len(self._actuator_names) == self.n
        self._validate_and_map()

        self.bound = np.full(self.n, clip_c, dtype=np.float32)
        self.bound[-1] = finger_bound_m
        self.clip_c = clip_c
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32).copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32).copy()
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

        # Reward / success (privilege — NEVER in Actor obs)
        self.z_lift_abs = float(self.contract.get("z_lift_abs", 0.135))
        self.success_hold_steps = int(self.contract.get("success_hold_steps", 20))  # H
        self.lambda_delta = float(self.contract.get("lambda_delta", 0.01))
        self.R_term = float(self.contract.get("R_term", 1.0))
        self.t_max = int(self.contract.get("t_max", 300))
        self.cube_rest_z = float(self.contract.get("cube_rest_z", 0.055))
        self.drop_margin = float(self.contract.get("drop_margin", 0.02))
        self.drop_z_abs = self.cube_rest_z + self.drop_margin
        self._cube_body = str(self.contract.get("cube_body", "red_cube"))
        self._cube_qposadr = self._resolve_cube_freejoint_qposadr(self._cube_body)
        self._joint_lo = np.array(
            [self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn), 0]
             for jn in self._joint_names],
            dtype=np.float64,
        )
        self._joint_hi = np.array(
            [self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn), 1]
             for jn in self._joint_names],
            dtype=np.float64,
        )
        self._reset_success_state()

    def _validate_and_map(self) -> None:
        if self.model.nu != self.n:
            raise ValueError(f"model.nu={self.model.nu} != contract n_dof={self.n}")

        model_act = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        if model_act == self._actuator_names:
            self._act_perm = None
        else:
            name_to_i = {n: i for i, n in enumerate(model_act)}
            missing = [n for n in self._actuator_names if n not in name_to_i]
            if missing:
                raise ValueError(f"actuators missing in model: {missing}; have {model_act}")
            self._act_perm = np.array(
                [name_to_i[n] for n in self._actuator_names], dtype=np.int32
            )

        self._qpos_adr = np.zeros(self.n, dtype=np.int32)
        self._dof_adr = np.zeros(self.n, dtype=np.int32)
        for i, jn in enumerate(self._joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise ValueError(f"joint {jn!r} not in model")
            self._qpos_adr[i] = self.model.jnt_qposadr[jid]
            self._dof_adr[i] = self.model.jnt_dofadr[jid]

        # Optional equality-coupled joints (e.g. finger_right) — not in Actor obs
        self._coupled_qposadr: dict[str, int] = {}
        for jn in self._equality_coupled:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self._coupled_qposadr[jn] = int(self.model.jnt_qposadr[jid])

    def _resolve_home_key(self) -> int:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid < 0:
            if self.model.nkey > 0:
                return 0
            raise ValueError("no home keyframe in model")
        return int(kid)

    def _resolve_cube_freejoint_qposadr(self, body_name: str) -> int:
        """World-z of freejoint qpos for body (freejoint qpos = [x,y,z,qw,qx,qy,qz])."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"cube body {body_name!r} not in model")
        jntadr = int(self.model.body_jntadr[bid])
        jntnum = int(self.model.body_jntnum[bid])
        if jntnum < 1:
            raise ValueError(f"body {body_name!r} has no joint")
        jid = jntadr  # first joint of body
        if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(
                f"body {body_name!r} joint0 type={int(self.model.jnt_type[jid])} "
                f"expected FREE={int(mujoco.mjtJoint.mjJNT_FREE)}"
            )
        return int(self.model.jnt_qposadr[jid])

    def _reset_success_state(self) -> None:
        self.held_steps = 0
        self.ever_lift = False
        self.success = False
        self.failure_mode = "none"
        self._t = 0
        self.cube_z = float(self.cube_rest_z)

    def _read_cube_z(self) -> float:
        # freejoint qpos layout: x, y, z, qw, qx, qy, qz
        return float(self.data.qpos[self._cube_qposadr + 2])

    def _check_unsafe(self) -> bool:
        """Minimal: NaN/Inf in state/ctrl, or q far outside joint hard range."""
        if not np.isfinite(self.q).all() or not np.isfinite(self.qdot).all():
            return True
        if not np.isfinite(self.data.ctrl).all():
            return True
        # MuJoCo limited joints soft-overshoot slightly; only flag gross violations
        margin = 0.05
        if np.any(self.q < self._joint_lo - margin) or np.any(self.q > self._joint_hi + margin):
            return True
        return False

    def _update_success(self) -> bool:
        """Update held_steps / ever_lift / success / failure_mode from cube_z.
        Returns success_just_now (True only on the step success first becomes True).
        """
        self.cube_z = self._read_cube_z()
        success_just_now = False
        if self.cube_z >= self.z_lift_abs:
            self.held_steps += 1
            self.ever_lift = True
        else:
            self.held_steps = 0
            # drop: ever lifted then fell near rest before success
            if (
                self.ever_lift
                and not self.success
                and self.cube_z < self.drop_z_abs
                and self.failure_mode == "none"
            ):
                self.failure_mode = "drop"

        if (not self.success) and (self.cube_z >= self.z_lift_abs) and (
            self.held_steps >= self.success_hold_steps
        ):
            self.success = True
            success_just_now = True
            # success clears a prior drop label if somehow set same-step (shouldn't)
            if self.failure_mode == "drop":
                self.failure_mode = "none"

        if self.failure_mode == "none" and self._check_unsafe():
            self.failure_mode = "unsafe"

        return success_just_now

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
            "reward_version": self.reward_version,
            "clip_c": float(self.clip_c),
            "chunk_refresh": bool(chunk_refresh),
            "chunk_stale": bool(chunk_stale),
            "n_substeps": self.n_substeps,
            "model_path": str(self.model_path),
            "mujoco_version": mujoco.__version__,
            # privilege success extras (NOT in Actor obs)
            "cube_z": float(self.cube_z),
            "success": bool(self.success),
            "ever_lift": bool(self.ever_lift),
            "held_steps": int(self.held_steps),
            "failure_mode": str(self.failure_mode),
            "t": int(self._t),
            "z_lift_abs": float(self.z_lift_abs),
            "success_hold_steps": int(self.success_hold_steps),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
        self.base.reset(self._seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self._read_state()
        self._reset_success_state()
        self.cube_z = self._read_cube_z()  # calibrate rest after keyframe
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
            "cube_z": float(self.cube_z),
            "success": False,
            "ever_lift": False,
            "held_steps": 0,
            "failure_mode": "none",
            "reward_version": self.reward_version,
            "z_lift_abs": float(self.z_lift_abs),
            "success_hold_steps": int(self.success_hold_steps),
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
        self._t += 1

        chunk_refresh = False
        chunk_stale = False
        self.phase += 1
        if self.phase >= self.chunk_size:
            self.chunk = self.base.infer_chunk({"q": self.q})
            self.phase = 0
            self.chunk_idx += 1
            chunk_refresh = True
        self.a_base = self.chunk[self.phase].copy()

        success_just_now = self._update_success()
        # r = -λ‖δ‖² + 1{success_just_now}·R_term  (no w_z·cube_z shaping)
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
            "equality_coupled": self._equality_coupled,
            "act_perm": None if self._act_perm is None else self._act_perm.tolist(),
            "ctrl_lo": self.ctrl_lo.tolist(),
            "ctrl_hi": self.ctrl_hi.tolist(),
        }
