# ManiSkill3 (3.0.1) ReBot B601-DM — interface draft

**Needs NVIDIA GPU + Vulkan.** On the box, SAPIEN fails at URDF load (`failed to find a rendering device`), so nothing here has created an env.

- `ReBot_Arm_DM_ms3.urdf`: copy of the vendor URDF (commit 7456d8b) with `package://` mesh paths rewritten to relative paths. The vendor file is unchanged.
- `rebot_b601_agent.py`: `ReBotB601DM(BaseAgent)`, uid `rebot_b601_dm`. It defines `pd_joint_pos` = arm (6, rad, `PDJointPosControllerConfig`) + gripper (1, m, `PDJointPosMimicControllerConfig`, `finger_right = -finger_left`). The action is 7-dim. Import and registration were checked on CPU in `.venv-ms3` (py3.11).
- Contract: `openarm_residual_gate/contract_maniskill_rebot_b601_dm.json`, with sections `sim` / `vla` (verbatim fragment) / `residual` (placeholders).
- Static check: `python3 scripts/check_maniskill_rebot_static.py` (log in `logs/maniskill_rebot_static/`).

## Still to do (4090 smoke)
1. `python -m mani_skill.examples.demo_random_action -e PickCube-v1` to confirm Vulkan works.
2. Load the agent and check the active-joint order (expected joint1..6, finger_left, finger_right), the rest keyframe, and mimic tracking.
3. Check the collision meshes of the thin fingers, then tune PD gains and gripper force.
4. Measure the TCP offset from `end_link`.
5. Write `PickCubeReBot-v1` (a subclass like `PickCubeSO100-v1`) with a cube size and workspace that fit the robot, and check that `evaluate()["success"]` behaves correctly.
6. Overhead camera pose. The wrist camera stays pending.
7. Measure throughput (num_envs 1/64/512) and run the delta=0 gate.
8. Pending decisions: base_source (user), stats_id + gripper deg endpoints + image keys (VLA), gripper clip_c + lambda (Residual RL).
