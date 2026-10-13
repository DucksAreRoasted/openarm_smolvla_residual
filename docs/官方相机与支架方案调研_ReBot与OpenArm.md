# 官方相机布局与支架调研（ReBot B601-DM 单臂 → OpenArm 双臂）

日期：2026-10-13。原则：**布局一旦定下就永久固定，优先官方方案**。图例：✅ 已核实（附来源/本地文件）；⚠️ 未核实/推断；❌ 官方缺失。
本地浅克隆（无 LFS）：`vendor/rebot_devarm_hw`（Seeed-Projects/reBot-DevArm）、`vendor/openarm_hw`（enactic/openarm 文档站），commit 记录在 `vendor/VERSIONS.txt`。现有契约/代码未改动。

---
## 1. ReBot Arm B601-DM（单臂）

### 1.1 官方相机型号与角色
| 项 | 结论 | 状态 / 来源 |
|---|---|---|
| 官方腕部相机支架覆盖的型号 | 32×32 UVC、RealSense **D405**（与 Gemini 305 共用）、D435i、Orbbec Gemini 2（与 D435 共用）、D455f | ✅ `hardware/camera-mounts/README.md` |
| 支架共用性 | “These four wrist camera mounts are shared by **B601-RS and B601-DM**” | ✅ 同上 |
| DM 硬件 BOM 中列出的“Compatible Camera Mount” | 只列 **Orbbec Gemini2 → `D435_Gemini2_Mount.step`**（Bambu ABS，0.4 喷嘴，0.2 层高，30% 填充，数量 1） | ✅ `hardware/reBot_B601_DM/readme.md` §📷 |
| 官方仿真/数字孪生中的腕部相机 | B601-RS 浏览器数字孪生：“view the overhead and **D405 wrist** cameras” | ✅ 主 README L90 |
| 官方 DM MJCF 相机 | 只有 `overhead_rgb`（`pos="0.42 0 0.86" xyaxes="1 0 0 0 1 0" fovy="50"`），**无腕部相机** | ✅（见 `docs/ReBot_B601_DM_Survey.md` §4.3） |
| 官方采集用外部相机支架 | `data-collection-camera-mounts/`：V1（4 个 3D 打印支架 + 适配件，含 `mount-1-rebot-adapter.stp`），V2 铝型材支架整装 `data-collection-camera-mount-v2.stp` | ✅ 但 README 原话：“completed assemblies and camera compatibility have not been verified”；V2 “supplied locally by the maintainer”，**无相机型号、无安装位置、无 BOM** ⚠️ |
| 官方 LeRobot 录制分辨率/fps | **官方未给出固定相机配置**；LeRobot `rebot_b601_follower` 相机字典由用户提供 | ⚠️（社区数据集均为 30 fps，见 §3） |
| 官方采购链接（README 硬件推荐表） | RealSense D405：https://www.seeedstudio.com/RealSense-D405-3D-Camera-p-6758.html ；D435i：https://www.seeedstudio.com/Intel-RealSense-Depth-Camera-D435i-p-4423.html ；Gemini 2：https://www.seeedstudio.com/Orbbec-Gemini-2-3D-Camera-p-6464.html ；ET-S231 120° USB：https://www.seeedstudio.com/ET-S231-120-USB-Camera-p-6683.html | ✅ 主 README L208–215 |

### 1.2 支架逐件清单
| 部件 | 文件（CAD） | 仿真网格 / URDF | 用途 / 安装位置 | 固定方式 |
|---|---|---|---|---|
| **D405_305_Mount** | https://github.com/Seeed-Projects/reBot-DevArm/blob/main/hardware/camera-mounts/b601-camera-mounts/D405_305_Mount.step | `Rebot_Arm_description/Camera/meshes/mounts/D405_305_Mount.stl`，`Camera/urdf/d405.urdf` | 腕部（eye-in-hand），挂在末端 `gripper_end` 参考系 | 弧形夹持底座卡在腕部 + 相机底部螺钉（`camera_bottom_screw_frame`）；**螺钉规格/扭矩官方未给** ⚠️；设计说明：https://github.com/bowenszhu/rebot-b601-rs-d405-wrist-mount/tree/v1.0.0 |
| D435_Gemini2_Mount | …/b601-camera-mounts/D435_Gemini2_Mount.step | `meshes/mounts/D435_Gemini2_Mount.stl`，`urdf/d435i.urdf`、`urdf/gemini2.urdf` | 腕部 | 同上 |
| D455f_Mount | …/b601-camera-mounts/D455f_Mount.step | 无 URDF | 腕部（推断）⚠️ | 未说明 |
| UVC32_mount | …/b601-camera-mounts/UVC32_mount.step | `meshes/mounts/UVC32_mount.stl`，`urdf/uvc32.urdf` | 腕部；28×28 mm 孔距（CAD 提取） | 同上 |
| 采集外部支架 V1 mount-1 + rebot adapter | …/data-collection-camera-mounts/V1/mount-1.stp、mount-1-rebot-adapter.stp | 无 | 桌面外部相机（推断）⚠️ | 无装配说明 |
| 采集外部支架 V2（铝型材） | …/data-collection-camera-mounts/data-collection-camera-mount-v2.stp | 无 | 桌面外部相机（推断）⚠️ | 型材+打印件，无 BOM |

注意：README 明确 “STL files, print settings, and a hardware bill of materials are not included” —— 打印需自己从 STEP 导出（`Camera/meshes/mounts/*.stl` 是 rebot_visual_grasp 的转换版，可直接用于仿真，打印前核对尺寸）。

### 1.3 腕部 D405 外参（官方 URDF，可直接用）✅
`Camera/README.md` 安装参数表（相对相机装配根 `gripper_end`，单位 m / rad）：
| 型号 | 支架 XYZ | 支架 RPY | 相机（底部螺钉系）XYZ | 相机 RPY |
|---|---|---|---|---|
| D405 | -0.1201 0.0003 0.0507 | 1.5827 0.0024 1.5515 | -0.0003 0.0443 -0.0093 | -0.0225 -1.0448 -1.5460 |
| D435i | -0.1201 0.0003 0.0450 | 1.5827 0.0024 1.5515 | -0.0003 0.0445 -0.0095 | 0.0126 -1.0489 -1.5981 |

本地沿 `d405.urdf` 链计算（本调研脚本，未经实物标定）：`camera_color_optical_frame` 相对 `gripper_end`：pos ≈ (-0.1087, 0.0091, 0.1076) m，光轴(+z_opt) ≈ (0.859, -0.005, -0.512)。
**关键坑**：RS 的 `gripper_end` = `link6` + `xyz 0 0 0.16621, rpy 3.1416 -1.5708 0`；DM 的 `end_link` = `link6` + `xyz 0 0 0.15539, rpy 0 -1.5708 3.1415`。两者**不相同**。README 原话：DM 需“用固定 joint 连接到 `end_link`，在该 joint 中设置安装参考系变换”——**官方未给出 DM 的这个变换值** ❌。
推导步骤：(1) 支架夹在 link6/腕部同一物理位置（RS/DM 共用支架和腕部件，⚠️需在 `reBot_B601_DM_v1.1_20260625.step` 中核对）；(2) 求 `T_end_link←gripper_end = T(link6→end_link)^-1 · T(link6→gripper_end)`（RS/DM 的 link6 若几何一致则成立）；(3) 在 STEP 中把 D405_305_Mount 装到 DM 腕部，量取支架夹持中心相对 link6 的位姿校验；(4) 实物上用棋盘格手眼标定（eye-in-hand）最终确认。

### 1.4 ManiSkill3 复现
- 腕部：`CameraConfig(uid="wrist", mount=robot.links_map["end_link"], pose=sapien.Pose(p, q))`，p/q = 上述 color optical 链 × 步骤(2) 的变换；注意 SAPIEN 相机系是 x 前、y 左、z 上，需把 optical(z 前, x 右, y 下) 右乘固定旋转。D405 彩色 FOV 官方规格约 87°×58°（⚠️ 未在本调研中从 datasheet 核实，请以 Intel 规格表为准）。
- 外部：官方仅有 MJCF `overhead_rgb`（fixed pose, fovy 50）。ManiSkill 用 `CameraConfig(uid="overhead", pose=sapien_utils.look_at(...)/直接由 pos+xyaxes 转换, fov=deg2rad(50))`，挂 world（`mount=None`）。
- **硬件上 overhead 没有官方支架/高度定义**：MJCF 位置 0.86 m 高、正下视；实物需用 V2 型材支架或自制龙门复现该位姿 ⚠️。

### 1.5 推荐固定布局（ReBot）
| key（`observation.images.*`） | 相机 | 支架 | 位置 | 分辨率/fps |
|---|---|---|---|---|
| `wrist` | **RealSense D405**（官方孪生、官方 URDF、购买链接齐全） | **D405_305_Mount.step**（官方） | 腕部，按 `d405.urdf` 外参 | 640×480 @30（⚠️项目约定，非官方） |
| `overhead` | 一台 RGB/RGB-D（建议 D435i，或任一 UVC；官方未指定型号 ❌） | data-collection-camera-mount-v2.stp（官方仓库，但未定义相机型号/位置 ⚠️）或自制龙门 | 世界固定，复刻 MJCF `overhead_rgb`：pos (0.42, 0, 0.86) 相对机械臂基座、垂直下视、fovy 50 | 640×480 @30 |
需购买/打印：D405 ×1、外部相机 ×1、USB3 线；打印 D405_305_Mount（ABS，0.4/0.2/30% 参照 BOM 同类件）；外部支架铝型材+打印件。

---
## 2. enactic OpenArm 双臂

### 2.1 两代“官方”布局（必须二选一）
| 代 | 相机 | 状态 / 来源 |
|---|---|---|
| **OpenArm 01 Release 1.1（v1 硬件）** | 胸部 **RealSense D435**（支撑立柱）+ 腕部 **RealSense D405** | ✅ Release 1.1 说明：“Added a modular camera mount base with standard mounts designed for Realsense D435 (chest camera) and D405 (wrist camera)” https://github.com/enactic/openarm/releases/tag/1.1 ；issue #307（closed）https://github.com/enactic/openarm/issues/307 |
| **OpenArm 2.0 + OpenArm Cell（当前官方，带标准化基准）** | 5 台：`camera_head_left/right`（**ZED-M 立体**，“head stereo camera is a ZED camera from Stereolabs”）、`camera_ceiling`、`camera_wrist_left/right`（**Arducam B0495**，内置于夹爪壳体，“one per arm”） | ✅ `website/docs/tutorial/setup.mdx`、`docs/hardware/openarm-2.0/gripper.mdx`；Cell 说明：“defining … lighting, cameras, and calibration procedures … as part of a unified system” |
| ceiling 相机型号 | udev 规则用 Arducam 序列号 `CELL1_CAM_CEILING` → 推断为 Arducam UVC，**具体型号未写** ⚠️ |
| Arducam B0495 | AR0234 全局快门，1920×1200@50 / 960×600@80（USB3），FOV 95°(D)×82°(H)×66°(V)，M12 3.6 mm | ✅ gripper.mdx |
| 推理/数据分辨率 | 推理示例：五路均 960×600；HF 仿真数据集：head 1280×720，其余 960×600，30 fps | ✅ inference.mdx；HF info.json |

### 2.2 支架/CAD 清单
| 部件 | 文件 | 位置 / 固定 | 状态 |
|---|---|---|---|
| `wrist-camera_base.step/.stl` | openarm_hardware Release 1.1.0 tarball `Attachments/`：https://github.com/enactic/openarm_hardware/releases/download/1.1.0/openarm-hardware-1.1.0.tar.gz | 腕部模块化底座 | ✅ 文件名已列出（tar 列表），**螺钉/孔位未查看** ⚠️ |
| `wrist-camera_D405.step/.stl` | 同上 | D405 适配件，装在 wrist base 上 | ✅ 同上 |
| `chest-camera_D435.step/.stl` | 同上 | D435 胸部，挂支撑立柱（v1） | ✅ 同上；立柱型号/高度 ⚠️ |
| OpenArm 2.0 头部相机选项 | `OpenArm 2.0/Options/OpenArm_2.0_w_Head_Camera.STEP`（2.0.0 tarball / Google Drive id `1mP0gDFgGaX6Ufobr8ocyg71ccsYgfBDw`） | 头部 ZED-M 立体（立柱为 MISUMI 铝型材：“easy to … attach cameras”） | ✅ 文件存在；内容未打开 ⚠️ |
| 腕部 Arducam | 集成在 2.0 夹爪壳体内（无独立支架） | — | ✅ |
| BOM | `OpenArm 2.0 BOM.xlsx`、`OpenArm Cell BOM.xlsx`（Drive: https://drive.google.com/drive/folders/1HzonRqvZ_1FZSTvcW09gT69bLr4alACN ；文件 id 见 openarm_hardware `dev/google-drive-files/file-ids.tsv`） | — | ✅ 存在；**相机行未读** ⚠️ |
| Cell 整体 | `OpenArm_Cell+OpenArm_2.0.STEP`、`OpenEvalCell_defeature.STEP` | ceiling 相机挂 cell 框架 | ✅ 存在 ⚠️ 未打开 |

### 2.3 仿真外参
v2 cell MJCF（vendor/openarm_mujoco，已在 `Camera_Layout_Survey.md` 摘录）✅：`camera_head_left/right` pos (0.223, ±0.0315, 1.45)，fovy 52，1280×720，worldbody；`camera_ceiling` (0.245,0,1.8)，fovy 66；`camera_wrist_*` 挂 `ee_base_link`，pos (0.023, ∓0.0182, -0.10236)，euler (0.3665, 0.2269, -1.5708)，fovy 66（=B0495 垂直 FOV 66°，一致）。Isaac Lab 无相机 ✅（旧文档）。v1 Release 1.1 的 D435/D405 **无官方 MJCF/URDF 外参** ❌，只能从 STEP 推导。
ManiSkill3：head/ceiling 用世界固定 `CameraConfig`（需同时复刻 cell 桌面 z≈1.00、lifter 安装 0.185,0,1.34）；wrist 用 `mount=links_map["openarm_{l,r}_ee_base_link"]`（需 v2 机器人模型）。

### 2.4 推荐固定布局（OpenArm 双臂）
**采用 OpenArm 2.0 + Cell 五相机布局为官方参考**（理由：唯一同时有硬件、MJCF 外参、HF 官方数据集和推理示例的布局）。keys：`head_left, head_right, ceiling, wrist_left, wrist_right`（官方训练示例过滤掉 `ceiling`、`head_right`，最小子集 = `head_left + wrist_left + wrist_right`）。需购买：ZED-M ×1、Arducam B0495 ×2（2.0 夹爪自带）、ceiling 相机（型号查 Cell BOM）、Cell 框架。
若硬件是 **v1**：只能用 Release 1.1 的 `chest_D435 + 2× wrist_D405`（打印 chest-camera_D435、wrist-camera_base ×2、wrist-camera_D405 ×2），但该布局**没有官方仿真外参**，需从 STEP 推导 + 标定。

---
## 3. 公开 HF 数据集对照（meta/info.json，2026-10-13 抓取）
| 数据集 | robot_type | ep | fps | 图像键（分辨率） | 相机型号/安装 | 符合官方布局？ |
|---|---|---|---|---|---|---|
| LeWAM/b601_pusht_pick_and_place | rebot_b601_follower | 200 | 30 | `zed_left`,`zed_right` 672×376；`wrist_cam` 640×360 | 外部 ZED 立体 + 腕部（型号/支架未写） | ❌ |
| binhpham/portal-rebot-pick-microcontroller | seeed_b601_dm_follower | 50 | 30 | `arm_camera`,`overhead_camera` 640×480 | 卡片：“arm-mounted RGB”、“overhead RGB”；型号未写 | ⚠️ 角色一致（wrist+overhead），型号/支架/位姿未知 → 不能视为匹配 |
| SieloRobotics/rebot_pick_place_20261005_170722 | rebot_b601_follower | 15 | 30 | `wrist`,`top` 640×480 | 未写 | ⚠️ 同上 |
| adrfm/pick_place_b601_v2 | seeed_b601_rs_follower（**RS 非 DM**） | 10 | 30 | `side`,`wrist` 640×480 | 未写 | ❌（side 非 overhead） |
| openrob-furg/openarm-single-pick-place-lego | openarm_follower | 200 | 30 | `cam_chest`,`right_cam_wrist` 640×480 | **D435i 胸部 + D405 右腕** | ✅ 接近 Release 1.1（v1，D435i≈D435；单臂；是否用官方打印件未写 ⚠️） |
| enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot | openarm_bimanual | 30 | 30 | `ceiling` 960×600、`head_left/right` 1280×720、`wrist_left/right` 960×600 | **仿真** v2 cell | ✅ 官方 2.0 Cell 布局（但仅仿真） |
（另有 `enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot-past`；enactic 账号下无真实硬件数据集。）
**结论：ReBot 没有任何公开数据集可确认符合官方 D405 支架布局 → 真机演示必须按固定布局重新采集。** OpenArm：2.0 Cell 仅有官方仿真数据；v1 方案下唯一接近的是 openrob-furg（单臂）。

---
## 4. 缺口 / 未核实
1. ReBot DM `end_link` ↔ 相机根 `gripper_end` 的固定变换官方未给 ❌（§1.3 步骤推导）。
2. ReBot 外部（overhead）相机：官方无型号、无实物位置；V2 支架无 BOM/装配说明 ⚠️。
3. 所有支架螺钉规格、扭矩未公开 ⚠️；ReBot 官方不提供可打印 STL（需从 STEP 导出）。
4. OpenArm ceiling 相机型号、ZED-M/ceiling 的 BOM 行与采购链接：需打开 Cell / 2.0 BOM.xlsx ⚠️。
5. OpenArm v1 chest D435 立柱高度与外参：仅 STEP，无数值 ❌。
6. 官方 LeRobot 录制分辨率：两家都没有写死的 LeRobot camera config ⚠️。

## 5. 下一步
1. 用户确认 OpenArm 硬件代（v1 → Release 1.1 chest+wrist；2.0 → Cell 五相机）。
2. 下载 `reBot_B601_DM_v1.1_20260625.step` + `D405_305_Mount.step`（FreeCAD/cadquery）求 DM `end_link→gripper_end` 变换，写入新的 ManiSkill CameraConfig（新文件，不改现有契约）。
3. 下载 OpenArm 2.0 / Cell BOM.xlsx（Drive）补齐 ceiling 型号与采购链接。
4. 打印/采购后做手眼标定，按固定 keys（ReBot: `wrist`,`overhead`）重采数据。
