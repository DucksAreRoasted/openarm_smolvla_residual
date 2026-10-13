"""PickCubeReBot-v1 (Cube Lift) for ReBot B601-DM, ManiSkill3 3.0.1. NEW file, no contract changed.

Cameras come from sim/maniskill_cameras/camera_config_v0.json (uids wrist / overhead).
PLACEHOLDERS (to verify on the 4090 smoke, flagged in info):
  cube_half_size, cube_spawn_center/half_size, lift_height (success threshold) -> not official.
  Camera fov for wrist (D405) -> null in config, fallback fovy 58 deg is a placeholder.
Privileged state (cube pose etc.) is only exposed under obs_mode containing "state" / in info,
for critic & reward; the residual actor must use [q, qdot, a_base, phase] only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebot_b601_agent import ReBotB601DM  # noqa: E402,F401  (registers agent)

_ROOT = Path(__file__).resolve().parents[3]
CAM_CFG = json.loads((_ROOT / "sim/maniskill_cameras/camera_config_v0.json").read_text())["rebot_b601_dm"]
WRIST_FOVY_PLACEHOLDER_DEG = 58.0  # PLACEHOLDER until D405 spec verified


def _overhead_pose(pos):
    # MJCF xyaxes "1 0 0 0 1 0": image right=+x, image up=+y, looks -z.
    # SAPIEN cam frame: x forward, y left, z up -> columns [(0,0,-1),(-1,0,0),(0,1,0)].
    m = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
    q = R.from_matrix(m).as_quat()
    return sapien.Pose(p=pos, q=[q[3], q[0], q[1], q[2]])


@register_env("PickCubeReBot-v1", max_episode_steps=300)
class PickCubeReBotEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["rebot_b601_dm"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    cube_half_size = 0.02            # PLACEHOLDER
    cube_spawn_center = (0.30, 0.0)  # PLACEHOLDER (overhead at x=0.42)
    cube_spawn_half_size = 0.05      # PLACEHOLDER
    lift_height = 0.05               # PLACEHOLDER (smoke only); final h = lift_h_mult * 2*cube_half_size
    lift_h_mult = 1.25               # confirmed 20261013 by weiqiang: h=5cm
    hold_steps_k = 10                # confirmed 20261013 by weiqiang (0.5s@20Hz)
    grasp_min_force = 0.5            # PLACEHOLDER contact-force threshold (N)

    def __init__(self, *args, robot_uids="rebot_b601_dm", sim_freq=100, control_freq=20, **kwargs):
        from mani_skill.utils.structs.types import SimConfig
        kwargs.setdefault("sim_config", SimConfig(sim_freq=sim_freq, control_freq=control_freq))
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        w, o = CAM_CFG["wrist"], CAM_CFG["overhead"]
        wfov = np.deg2rad(w["fov_rad"] and np.rad2deg(w["fov_rad"]) or WRIST_FOVY_PLACEHOLDER_DEG)
        wrist = CameraConfig(
            uid="wrist", pose=sapien.Pose(p=w["pose_p"], q=w["pose_q_wxyz"]),
            width=w["width"], height=w["height"], fov=float(wfov), near=0.01, far=10,
            mount=self.agent.robot.links_map[w["mount_link"]],
        )
        overhead = CameraConfig(
            uid="overhead", pose=_overhead_pose(o["pos"]), width=o["width"], height=o["height"],
            fov=float(np.deg2rad(o["fovy_deg"])), near=0.01, far=10,
        )
        return [wrist, overhead]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube",
            initial_pose=sapien.Pose(p=[*self.cube_spawn_center, self.cube_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.as_tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32).repeat(b, 1)
            self.agent.reset(qpos)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            if hasattr(self, "_hold"):
                self._hold[env_idx] = 0

    def _lift_h(self):
        return self.lift_h_mult * 2 * self.cube_half_size if self.lift_h_mult else self.lift_height

    def is_grasping(self, obj, min_force=None):
        # both fingers in contact with obj (contact-force based, like ManiSkill panda)
        mf = self.grasp_min_force if min_force is None else min_force
        l = self.scene.get_pairwise_contact_forces(self.agent.finger_left_link, obj)
        r = self.scene.get_pairwise_contact_forces(self.agent.finger_right_link, obj)
        return (torch.linalg.norm(l, axis=1) >= mf) & (torch.linalg.norm(r, axis=1) >= mf)

    def evaluate(self):
        z = self.cube.pose.p[:, 2]
        lift_height = z - self.cube_half_size  # bottom-face rise vs. initial (on ground)
        lifted = lift_height >= self._lift_h()
        is_grasped = self.is_grasping(self.cube)
        fl = torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.agent.finger_left_link, self.cube), axis=1)
        fr = torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.agent.finger_right_link, self.cube), axis=1)
        ok = lifted & is_grasped
        if not hasattr(self, "_hold") or self._hold.shape != ok.shape:
            self._hold = torch.zeros_like(ok, dtype=torch.int32)
        self._hold = torch.where(ok, self._hold + 1, torch.zeros_like(self._hold))
        return dict(success=self._hold >= self.hold_steps_k, is_grasped=is_grasped, lifted=lifted,
                    lift_height=lift_height, hold_count=self._hold.clone(),
                    contact_force_left=fl, contact_force_right=fr)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if "state" in self.obs_mode:  # privileged: critic/reward only
            obs.update(obj_pose=self.cube.pose.raw_pose)
        return obs

    def compute_sparse_reward(self, obs, action, info):
        return info["success"].float()
