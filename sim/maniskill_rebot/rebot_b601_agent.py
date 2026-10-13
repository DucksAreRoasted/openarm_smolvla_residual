"""ManiSkill3 (v3.0.1) agent DRAFT for ReBot B601-DM.

Symbols verified by import against mani_skill==3.0.1 (CPU venv .venv-ms3 / /tmp/ms311):
  BaseAgent, Keyframe, register_agent, PDJointPosControllerConfig,
  PDJointPosMimicControllerConfig (mimic dict format copied from panda.py / fetch.py).
NOT verified: env creation / articulation load (no GPU/Vulkan on box), PD gains, rest pose,
collision quality of finger meshes, tcp offset. Run the 4090 smoke before trusting any of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent

_HERE = Path(__file__).resolve().parent
URDF_PATH = str(_HERE / "ReBot_Arm_DM_ms3.urdf")

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_CONTROL_JOINT = "finger_left"     # 0..0.05 m, the only commanded gripper dof
GRIPPER_MIMIC_JOINT = "finger_right"      # -0.05..0 m, = -1 * finger_left (URDF <mimic>)
TCP_LINK_NAME = "end_link"                # origin = fingertip plane; approach axis = end_link +x (from finger mesh bbox x in [-0.089, 0])


@register_agent()
class ReBotB601DM(BaseAgent):
    uid = "rebot_b601_dm"
    urdf_path = URDF_PATH
    fix_root_link = True
    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link=dict(
            finger_left_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            finger_right_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    # qpos order = SAPIEN active-joint order (expected joint1..6, finger_left, finger_right;
    # VERIFY on 4090 via robot.get_active_joints()). All-zero is inside every URDF limit
    # (joint2/joint3 upper=0). Placeholder: align with FakeBase/real home later.
    keyframes = dict(
        # "ready" v2: retracted to x~0.14 m, h~0.20 m, approach tilted 0.35 rad toward +x so overhead sees all 9 spawn cells; j6=3.10 (not 3.14) to keep margin from URDF limit
        # over cube spawn center (0.30, 0). Solved by CPU FK/IK from URDF; verify on GPU smoke.
        rest=Keyframe(qpos=np.array([0.0014, -0.7907, -1.0702, 1.5003, 0.0009, 3.10, 0.02, -0.02]), pose=sapien.Pose()),
        zero=Keyframe(qpos=np.zeros(8), pose=sapien.Pose()),
    )

    arm_joint_names = ARM_JOINT_NAMES
    gripper_joint_names = [GRIPPER_CONTROL_JOINT, GRIPPER_MIMIC_JOINT]
    ee_link_name = TCP_LINK_NAME

    # Gains: so100-style starting point, UNIDENTIFIED. URDF effort: j1-3=27, j4-6=7, fingers=8.
    arm_stiffness = [1e3] * 6
    arm_damping = [1e2] * 6
    arm_force_limit = [27, 27, 27, 7, 7, 7]
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 8

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,  # None -> use URDF joint limits
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,  # absolute rad targets (a_base + clip(delta))
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=0.05,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            mimic={GRIPPER_MIMIC_JOINT: {"joint": GRIPPER_CONTROL_JOINT, "multiplier": -1.0, "offset": 0.0}},
        )
        # action = [arm(6, rad), gripper(1, m)] -> 7-dim
        return dict(pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos))

    def _after_init(self):
        from mani_skill.utils import sapien_utils

        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self.finger_left_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "finger_left_link")
        self.finger_right_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "finger_right_link")
        # Finger meshes are L-shaped (racks cross the centreline, y in [-0.031, 0.059] / [-0.059, 0.031]);
        # their convex collision hulls overlap each other and the gripper base on link6 at every opening,
        # which jammed the fingers at +-0.021 m. Fingers are children of end_link (fixed to link6), so
        # SAPIEN's parent-child filter does not cover link6. Put fingers + link6 + end_link in one
        # ignore group (group 2, bit 4); finger-vs-cube/world contacts are unaffected.
        for name in ("finger_left_link", "finger_right_link", "link6", "end_link"):
            self.robot.links_map[name].set_collision_group_bit(group=2, bit_idx=4, bit=1)

    @property
    def tcp_pose(self):
        return self.tcp.pose  # TODO: add fingertip offset once measured

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return (qvel.abs() <= threshold).all(dim=1)
