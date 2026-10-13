#!/usr/bin/env python3
"""Fetch pinned ReBot assets and regenerate planning STL files locally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify existing assets without downloads or writes")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/rebot_asset_manifest.json").read_text())
    vendor = ROOT / "vendor/rebot_arm_digitaltwin_dm"
    if not args.check_only:
        if not vendor.exists():
            subprocess.run(["git", "clone", manifest["repository"], str(vendor)], check=True)
            subprocess.run(["git", "-C", str(vendor), "checkout", "--detach", manifest["revision"]], check=True)
        meshes = vendor / "reBotArm_ros2_DM/src/rebotarm_bringup/description/meshes"
        planning = ROOT / "sim/maniskill_rebot/plan/meshes"
        import trimesh
        if trimesh.__version__ != manifest["trimesh_version"]:
            raise SystemExit(f"Install trimesh=={manifest['trimesh_version']} to reproduce the planning meshes")
        planning.mkdir(parents=True, exist_ok=True)
        for name in manifest["sha256"]:
            target = ROOT / name
            if target.parent != planning or target.exists():
                continue
            source_name = target.name.removesuffix(".convex.stl")
            mesh = trimesh.load_mesh(meshes / source_name)
            if target.name.endswith(".convex.stl"):
                mesh = mesh.convex_hull
            target.write_bytes(mesh.export(file_type="stl"))
        link = vendor / "reBotArm_ros2_DM/src/rebotarm_mujoco/meshes"
        if not link.exists() and not link.is_symlink():
            link.symlink_to("../rebotarm_bringup/description/meshes", target_is_directory=True)
    failures = []
    for name, expected in manifest["sha256"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise SystemExit("Missing or mismatched assets:\n" + "\n".join(failures))
    print(f"PASS: {len(manifest['sha256'])} asset SHA256 checks")


if __name__ == "__main__":
    main()
