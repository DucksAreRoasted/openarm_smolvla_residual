#!/usr/bin/env python3
"""EGL headless smoke: render each contract camera one frame → logs/camera_smoke/.

Locked layout: head + wrist, native 640×480 RGB, target 30 fps (smoke = one frame each).
Does NOT wire images into ResidualEnv Actor obs (Actor stays [q,qdot,a_base,phase]).
Not a full VLA / LeRobot observation pipeline — only proves cameras render non-black.

Usage (from repo root):
  source .venv/bin/activate
  export MUJOCO_GL=egl
  python3 scripts/render_cameras_smoke.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = _ROOT / "openarm_residual_gate" / "contract.json"
_OUT = _ROOT / "logs" / "camera_smoke"


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal RGB PNG writer (no Pillow dependency)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {rgb.shape} {rgb.dtype}")
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _body_label(model: mujoco.MjModel, body_id: int) -> str:
    if body_id == 0:
        return "worldbody(id=0)"
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name or f"body_{body_id}"


def main() -> int:
    contract = json.loads(_CONTRACT.read_text())
    scene = _ROOT / contract["model_xml_scene"]
    cams = contract.get("cameras") or []
    if not cams:
        print("ERROR: contract.cameras empty", file=sys.stderr)
        return 1
    if not contract.get("camera_in_env", False):
        print("WARN: camera_in_env is false; continuing render smoke anyway")

    expected_names = {"head", "wrist"}
    got_names = {c["name"] for c in cams}
    if got_names != expected_names:
        print(f"WARN: expected cameras {sorted(expected_names)}, got {sorted(got_names)}")

    _OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home < 0:
        raise RuntimeError("home keyframe missing")
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"mujoco={mujoco.__version__}")
    print(f"model={scene}")
    print(f"ncam={model.ncam} camera_fps={contract.get('camera_fps')} out={_OUT}")

    results = []
    renderer = None
    cur_size = None
    for cam in cams:
        name = cam["name"]
        width = int(cam["width"])
        height = int(cam["height"])
        lerobot_key = cam.get("lerobot_key", f"observation.images.{name}")

        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            print(f"FAIL camera {name!r} not in model", file=sys.stderr)
            return 1
        body_id = int(model.cam_bodyid[cid])
        body_name = _body_label(model, body_id)

        if cur_size != (width, height):
            if renderer is not None:
                renderer.close()
            renderer = mujoco.Renderer(model, height=height, width=width)
            cur_size = (width, height)

        renderer.update_scene(data, camera=name)
        rgb = renderer.render()  # HxWx3 uint8
        if rgb.shape != (height, width, 3):
            print(f"FAIL {name}: shape {rgb.shape} != ({height},{width},3)", file=sys.stderr)
            return 1

        mean = float(rgb.mean())
        if mean < 1.0:
            print(f"FAIL {name}: near-black frame mean={mean:.4f}", file=sys.stderr)
            return 1

        png_path = _OUT / f"{name}.png"
        npy_path = _OUT / f"{name}.npy"
        _write_png(png_path, rgb)
        np.save(npy_path, rgb)

        line = (
            f"  {name}: shape={tuple(rgb.shape)} dtype={rgb.dtype} mean={mean:.2f} "
            f"body={body_name} mount={cam.get('mount')} lerobot_key={lerobot_key} "
            f"png={png_path} npy={npy_path}"
        )
        print(line)
        results.append(
            {
                "name": name,
                "lerobot_key": lerobot_key,
                "shape": list(rgb.shape),
                "mean": mean,
                "body": body_name,
                "body_id": body_id,
                "width": width,
                "height": height,
                "fps": cam.get("fps", contract.get("camera_fps")),
                "png": str(png_path),
                "npy": str(npy_path),
            }
        )

    if renderer is not None:
        renderer.close()

    meta = _OUT / "smoke_meta.json"
    meta.write_text(
        json.dumps(
            {
                "mujoco": mujoco.__version__,
                "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                "model": str(scene),
                "camera_in_env": contract.get("camera_in_env"),
                "camera_in_actor_obs": contract.get("camera_in_actor_obs"),
                "camera_fps": contract.get("camera_fps"),
                "cameras": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"meta={meta}")
    print("PASS camera smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
