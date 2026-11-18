#!/usr/bin/env python3
"""Offscreen camera render benchmark for Atharv Nandi (no ROS).

Loads the same perception model as atharv_nandi/nodes/perception_node.py (farm world + head_cam / wrist_cam via
atharv_nandi/core/camera_rig.py), renders N frames of one camera at 160x120, 320x240 and 640x480,
and prints frames per second for an RGB pass and for the single RGB+depth pass the node uses.

* macOS host (native, Apple GPU through CGL):
      uv venv --python 3.12 ~/.venvs/nandi-gpu   # not Homebrew python3 (3.15): mujoco has no wheel for it
      uv pip install --python ~/.venvs/nandi-gpu/bin/python "mujoco>=3.3,<3.4" numpy
      ~/.venvs/nandi-gpu/bin/python tools/mac_gpu/render_bench.py
* Container (Mesa llvmpipe, software GL), from the repo root:
      docker run --rm -v "$PWD:/ws" -w /ws atharv-nandi-sim bash -c \\
        'Xvfb :9 -screen 0 640x480x24 & sleep 1; DISPLAY=:9 MUJOCO_GL=glfw python3 tools/mac_gpu/render_bench.py'

The run is capped (--max-seconds, default 45 s) so it stays a quick check. Results go in
tools/mac_gpu/README.md.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

if "MUJOCO_GL" not in os.environ:
    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"      # native macOS OpenGL -> Apple GPU
    elif not os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from atharv_nandi.core import camera_rig  # noqa: E402

DEFAULT_RES = "160x120,320x240,640x480"


def gl_renderer_name(cam: camera_rig.CameraRenderer) -> str:
    try:
        ctx = getattr(cam.renderer, "_gl_context", None)
        if ctx is not None:
            ctx.make_current()
        from OpenGL import GL  # PyOpenGL ships with the mujoco wheel's dependencies
        return f"{GL.glGetString(GL.GL_VENDOR).decode()} / {GL.glGetString(GL.GL_RENDERER).decode()}"
    except Exception as exc:  # pragma: no cover - informational only
        return f"unknown ({type(exc).__name__})"


def bench(fn, data, frames: int, deadline: float) -> float:
    for _ in range(3):  # warm-up: context, shader and texture upload
        fn(data)
    n, t0 = 0, time.perf_counter()
    while n < frames and time.perf_counter() < deadline:
        fn(data)
        n += 1
    return n / max(time.perf_counter() - t0, 1e-9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=200, help="frames per resolution and mode")
    ap.add_argument("--camera", default=camera_rig.HEAD_CAM, choices=camera_rig.CAMERA_NAMES)
    ap.add_argument("--res", default=DEFAULT_RES, help="comma-separated WxH list")
    ap.add_argument("--full-quality", action="store_true", help="keep shadows + reflections on")
    ap.add_argument("--max-seconds", type=float, default=45.0, help="overall time cap")
    ap.add_argument("--json", action="store_true", help="also print one JSON line with the results")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    model = camera_rig.build_perception_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    sizes = [tuple(int(v) for v in s.lower().split("x")) for s in args.res.split(",") if s.strip()]
    budget = args.max_seconds - (time.perf_counter() - t_start)
    per_mode = max(1.0, budget / (2 * len(sizes)))
    rows, gl_name = [], None
    for w, h in sizes:
        cam = camera_rig.CameraRenderer(model, args.camera, w, h, fast=not args.full_quality)
        gl_name = gl_name or gl_renderer_name(cam)
        rgb_fps = bench(cam.rgb, data, args.frames, time.perf_counter() + per_mode)
        rgbd_fps = bench(cam.rgbd, data, args.frames, time.perf_counter() + per_mode)
        cam.close()
        rows.append({"width": w, "height": h, "rgb_fps": round(rgb_fps, 1), "rgbd_fps": round(rgbd_fps, 1)})

    info = {
        "date": datetime.date.today().isoformat(),
        "host": f"{platform.system()} {platform.machine()} ({platform.processor() or 'n/a'})",
        "cpus": os.cpu_count(),
        "mujoco": mujoco.__version__,
        "gl_backend": os.environ.get("MUJOCO_GL", "default"),
        "gl_renderer": gl_name,
        "lp_num_threads": os.environ.get("LP_NUM_THREADS"),
        "camera": args.camera,
        "quality": "full" if args.full_quality else "fast (no shadows/reflections)",
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    print(f"# Atharv Nandi render bench {info['date']}: {info['host']}, {info['cpus']} CPUs, "
          f"mujoco {info['mujoco']}, MUJOCO_GL={info['gl_backend']}")
    print(f"# GL: {info['gl_renderer']}; camera {args.camera}; {info['quality']}; {info['wall_s']} s")
    print("| resolution | RGB fps | RGB+depth fps |")
    print("| --- | ---: | ---: |")
    for r in rows:
        print(f"| {r['width']}x{r['height']} | {r['rgb_fps']} | {r['rgbd_fps']} |")
    if args.json:
        print(json.dumps({"info": info, "results": rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
