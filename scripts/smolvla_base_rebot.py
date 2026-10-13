"""Frozen SmolVLA base for PickCubeReBot-v1 B0 eval / residual a_base. NEW file (2026-10-13).

Untested on GPU (box has no torch/lerobot). Written against lerobot 0.6.1 API:
  SmolVLAPolicy.from_pretrained(ckpt) + make_pre_post_processors(cfg, pretrained_path=ckpt).
Chunk scheduling: policy.select_action() keeps an internal queue; a new chunk is inferred when
the queue is empty, i.e. every n_action_steps control steps (VLA handbook: chunk_size=50,
n_action_steps=50). reset() clears the queue at every episode start.
Obs mapping must match the training data (convert_lerobot_rebot.py / gen_mp_demos_rebot.py):
  observation.state = qpos[:7] (joint1..6 + finger_left), float32
  wrist -> observation.images.camera1, overhead -> observation.images.camera2 (training rename_map)
  task = "pick up the cube and lift it"
Output: action (7,) = joint1..6 abs rad + finger_left m, sent unchanged to pd_joint_pos (delta=0).
"""
from __future__ import annotations
import time

TASK = "pick up the cube and lift it"
CAM_MAP = {"wrist": "observation.images.camera1", "overhead": "observation.images.camera2"}


class SmolVLABase:
    def __init__(self, ckpt, env, task=TASK, n_action_steps=None, device="cuda"):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.torch, self.env, self.task, self.device = torch, env, task, device
        self.policy = SmolVLAPolicy.from_pretrained(ckpt)
        if n_action_steps is not None:
            self.policy.config.n_action_steps = int(n_action_steps)
        self.policy.to(device).eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        try:
            from lerobot.policies.factory import make_pre_post_processors
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=ckpt)
        except Exception as e:  # no processors saved -> refuse instead of silently skipping normalization
            raise SystemExit(f"[smolvla] cannot load pre/post processors from {ckpt}: {e!r}")
        self.n_action_steps = int(self.policy.config.n_action_steps)
        self.t = 0; self.last_latency_ms = None

    def reset(self):
        self.policy.reset(); self.t = 0

    @property
    def chunk_idx(self):  # index of the chunk that produced the action of the *last* act() call
        return (self.t - 1) // self.n_action_steps

    @property
    def phase(self):  # position inside the current chunk, in [0, n_action_steps)
        return (self.t - 1) % self.n_action_steps

    def _batch(self, obs):
        torch = self.torch
        q = self.env.unwrapped.agent.robot.get_qpos()[..., :7]
        b = {"observation.state": q.to(self.device, torch.float32).reshape(1, 7), "task": [self.task]}
        for cam, key in CAM_MAP.items():
            x = obs["sensor_data"][cam]["rgb"]
            x = torch.as_tensor(x).to(self.device)
            if x.dim() == 3: x = x[None]
            b[key] = x.permute(0, 3, 1, 2).float() / 255.0  # (1,3,H,W) in [0,1]
        return b

    def act(self, obs):
        torch = self.torch
        t0 = time.time()
        with torch.inference_mode():
            a = self.post(self.policy.select_action(self.pre(self._batch(obs))))
        self.last_latency_ms = (time.time() - t0) * 1000.0
        self.t += 1
        a = a.reshape(1, -1)[:, :7]
        if not torch.isfinite(a).all():
            raise FloatingPointError(f"non-finite a_base at t={self.t}")
        return a.to(self.env.unwrapped.device) if hasattr(self.env.unwrapped, "device") else a
