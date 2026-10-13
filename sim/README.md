# Sim assets

## ReBot B601-DM (default M0–M2 residual gate)

- **Local overlay**: `rebot_b601/scene_residual.xml`
  - Content from vendor `rebotarm_b601_stl.xml` with `meshdir` retargeted to
    `vendor/rebot_arm_digitaltwin_dm/.../rebotarm_mujoco/meshes` (symlink → bringup meshes).
  - Adds `<actuator>` position ×7 (`joint1_ctrl`…`joint6_ctrl`, `finger_ctrl`),
    equality `finger_right = -finger_left`, and `home` keyframe.
  - **Vendor XML untouched** (upstream `nu=0`).
- **Cube**: body `red_cube` (freejoint box size 0.025) — prefer this over inventing a cube.
- **Cameras**:
  - Official `overhead_rgb` kept (pos=`0.42 0 0.86`, xyaxes=`1 0 0 0 1 0`, fovy=50) → contract name `overhead`, LeRobot `observation.images.overhead`.
  - **Wrist NOT invented** (no official MJCF wrist; hardware D405 pending). Key `observation.images.wrist` reserved in contract only.
- **DoF**: 7 controlled (`joint1`…`joint6` + `finger_left`); `nq=29` includes freejoint objects + equality-coupled `finger_right`.
- Contract: `../openarm_residual_gate/contract_rebot_b601_dm.json` (`control_hz=20`, `gripper_unit=m`).
- Survey: `../docs/ReBot_B601_DM_Survey.md` (vendor @ `7456d8b`).

## OpenArm v1 (archived path)

- **Canonical single-arm MJCF**: `../vendor/openarm_mujoco/v1/openarm.xml` (robot-only; **LOCAL** wrist camera under `openarm_hand`) or `../vendor/openarm_mujoco/v1/scene_single_arm.xml` (robot + floor + **head** camera; local overlay).
- **Do not** use upstream `v1/scene.xml` for residual gate — it includes **bimanual** `openarm_bimanual.xml`.
- Nested `<include>` from outside `v1/` breaks `meshdir="meshes"` resolution; keep scenes next to the meshes or load by absolute path from Python.
- Contract remains `../openarm_residual_gate/contract.json` (OpenArm; not overwritten by ReBot).
