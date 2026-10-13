#!/usr/bin/env python3
"""CPU-only static check for the ManiSkill3 ReBot B601-DM draft. No env creation."""
import hashlib, json, sys, xml.etree.ElementTree as ET
from pathlib import Path

R = Path(__file__).resolve().parents[1]
URDF = R / "sim/maniskill_rebot/ReBot_Arm_DM_ms3.urdf"
C = R / "openarm_residual_gate/contract_maniskill_rebot_b601_dm.json"
FRAG = R / "openarm_residual_gate/contract_maniskill_rebot_b601_dm.vla_fragment.json"
fails = []
def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok: fails.append(name)

root = ET.parse(URDF).getroot(); check("urdf_xml_parse", root.tag == "robot")
joints = {j.get("name"): j for j in root.findall("joint")}
exp = {f"joint{i}": "revolute" for i in range(1, 7)} | {"end_joint": "fixed", "finger_left": "prismatic", "finger_right": "prismatic"}
for n, t in exp.items():
    check(f"joint_type:{n}", n in joints and joints[n].get("type") == t)
lim = lambda n: [float(joints[n].find("limit").get(k)) for k in ("lower", "upper")]
c = json.loads(C.read_text()); s = c["sim"]
for n, l in s["joint_limits"].items():
    check(f"limit:{n}", lim(n) == l, f"urdf={lim(n)} contract={l}")
m = joints["finger_right"].find("mimic")
check("mimic", m is not None and m.get("joint") == "finger_left" and float(m.get("multiplier")) == -1 and float(m.get("offset", 0)) == 0)
meshes = [e.get("filename") for e in root.iter("mesh")]
missing = [f for f in meshes if f.startswith("package://") or not (URDF.parent / f).resolve().exists()]
check("meshes_exist_relative", not missing, f"n={len(meshes)} missing={missing[:3]}")
links = {l.get("name") for l in root.findall("link")}
check("tcp_link_exists", s["tcp_link"] in links)
check("action_dim==7", s["action_dim"] == 7 == len(s["joint_names"]) and c["residual"]["action_dim"] == 7)
check("joint_names==arm+gripper_control", s["joint_names"] == s["arm_joint_names"] + [s["gripper_control_joint"]])
check("mimic_not_in_action", s["gripper_mimic_joint"] not in s["joint_names"])
check("control_mode", s["control_mode"] == "pd_joint_pos")
check("freq_divisible", s["sim_freq_hz"] % s["control_freq_hz"] == 0)
check("units", s["units"] == {"arm": "rad", "gripper": "m"})
check("base_source_pending", s["base_source"] == "pending_user_decision")
check("wrist_not_invented", all(cam["pose"] is None for cam in s["cameras"] if cam["uid"] == "wrist"))
check("object_pose_not_in_actor", not s["object_pose_in_actor_obs"] and "object_pose" not in c["residual"]["actor_obs_keys"])
# VLA fragment: verbatim + joints consistent with URDF
frag = json.loads(FRAG.read_text())
check("vla_fragment_verbatim", c["vla"] == frag)
g = frag["adapter"]["gripper"]
check("vla_gripper_joint_in_urdf", g["urdf_joint"] in joints and g["urdf_joint"] == s["gripper_control_joint"])
check("vla_gripper_range==urdf", g["urdf_range_m"] == lim(g["urdf_joint"]))
check("vla_mimic_matches_urdf", "finger_right" in g["mimic"] and "-1" in g["mimic"])
check("vla_lerobot_dims==7", len(frag["lerobot_joint_names"]) == 7 == frag["lerobot_state_dim"] == len(s["joint_names"]))
check("vla_gripper_endpoints_still_pending", isinstance(g["lerobot_range_deg"], str) and "pending" in g["lerobot_range_deg"])
rc = c["residual"]["clip_c"]
check("residual_clip_split", rc["arm"]["schedule"] == [0.03, 0.05, 0.10] and rc["arm"]["unit"] == "rad" and rc["gripper"]["unit"] == "m")
# frozen assets
sys.path.insert(0, str(R / "openarm_residual_gate"))
from obs_keys_hash import compute_obs_keys_hash
for f, h in [("contract_rebot_b601_dm.json", "b8b5eef23fc3dd73"), ("contract_libero_spatial_v0.json", "d9807cfaf4a8701c")]:
    got = compute_obs_keys_hash(json.loads((R / "openarm_residual_gate" / f).read_text()))
    check(f"frozen_hash:{f}", got == h, got)
for f, h in [("contract.json", "c0a0faeba463612c"), ("contract_rebot_b601_dm.json", "33595b135204917d"), ("contract_libero_spatial_v0.json", "9757b1cb16202098")]:
    got = hashlib.sha256((R / "openarm_residual_gate" / f).read_bytes()).hexdigest()[:16]
    check(f"frozen_file_sha256:{f}", got == h, got)
print("OVERALL:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
