# Mac GPU lane: what the Apple GPU can and cannot do for Atharv Nandi

Short answer to "can we use the Mac GPU too?": **partly, and only natively on the macOS host.**
Docker Desktop on macOS runs containers in a Linux VM with no Metal/MPS passthrough, so nothing
inside the `atharv-nandi` container can reach the Apple GPU. The container renders with Mesa
llvmpipe, which is OpenGL on the CPU. The Apple GPU is only usable from a host Python process.

| Workload | Can the Mac GPU help? | Notes |
| --- | --- | --- |
| Camera rendering (`head_cam` / `wrist_cam`, offscreen MuJoCo) | **Yes, natively.** | MuJoCo's CGL backend (`MUJOCO_GL=cgl`) uses the Apple GPU. `render_bench.py` measures the gain. |
| Future vision-model inference (crop detector replacing `sim_truth_noise`) | **Yes, natively.** | Use PyTorch with the `mps` device on the host. |
| MuJoCo physics (`sim_runner`, `mj_step`) | **No.** | MuJoCo physics runs on the CPU only. The GPU does not change RTF. |
| RL training in the container (`docker/train.sh`, PPO + SB3) | **No.** | It runs inside the Linux VM, which has no GPU access. |
| RL training natively on the host with `device="mps"` | **UNVERIFIED, probably little.** | PPO with small MLP policies is bound by environment stepping on the CPU, so MPS usually gains little. Not measured here. |

## Benchmark

`render_bench.py` loads the same model as `atharv_nandi/nodes/perception_node.py`, using
`atharv_nandi/core/camera_rig.py` (farm world plus the two cameras). It renders `head_cam` at
160x120, 320x240 and 640x480 and prints frames per second for:

- an RGB pass;
- the single RGB+depth pass that the perception node uses.

It does not use ROS and needs only `mujoco` and `numpy`. The whole run is capped at 45 s.

```bash
# macOS host, native (Apple GPU via CGL). This is the only thing the lane runs on the Mac.
# Do not use the Homebrew `python3` here: it is Python 3.15, and mujoco 3.3.x ships macOS arm64
# wheels only for CPython 3.9 to 3.13, so pip would try (and fail) a source build. uv fetches a
# CPython 3.12 that has wheels for both mujoco and numpy.
uv venv --python 3.12 ~/.venvs/nandi-gpu
uv pip install --python ~/.venvs/nandi-gpu/bin/python "mujoco>=3.3,<3.4" numpy
# Without uv: /usr/bin/python3 (Xcode Command Line Tools, Python 3.9) also has wheels for both.
#   /usr/bin/python3 -m venv ~/.venvs/nandi-gpu && ~/.venvs/nandi-gpu/bin/pip install "mujoco>=3.3,<3.4" numpy
~/.venvs/nandi-gpu/bin/python tools/mac_gpu/render_bench.py --json

# Container (llvmpipe), from the repository root
docker run --rm -v "$PWD:/ws" -w /ws atharv-nandi-sim bash -c \
  'Xvfb :9 -screen 0 640x480x24 & sleep 1; DISPLAY=:9 MUJOCO_GL=glfw python3 tools/mac_gpu/render_bench.py --json'
```

By default shadows and reflections are off. The perception node renders the same way, because on
llvmpipe shadows and reflections triple the frame time. Pass `--full-quality` to keep them.

## Results

Every number below carries its device, run id and date, or says "not measured" (run-identity rule).

**Device A:** MacBook with an Apple M5 (10 cores, 24 GB). Docker Desktop VM: linux/aarch64,
8 vCPU, 16 GB. Container `atharv-nandi-sim` capped at `--cpus=2 --memory=2g`. Software: mujoco
3.3.7, Mesa llvmpipe (LLVM 15.0.7, 128 bits), `MUJOCO_GL=glfw` on Xvfb, fast quality, `head_cam`
at the home keyframe.

| Run id | Date | Device / backend | 160x120 RGB / RGB+D fps | 320x240 RGB / RGB+D fps | 640x480 RGB / RGB+D fps |
| --- | --- | --- | ---: | ---: | ---: |
| `pbench-20261117-c1` | 2026-11-17 | Device A, container, llvmpipe, default threads | 54.7 / 46.4 | 45.2 / 41.7 | 30.8 / 29.5 |
| `pbench-20261117-c2` | 2026-11-17 | Device A, container, llvmpipe, `LP_NUM_THREADS=1` (perception-node setting) | 35.6 / 35.8 | 27.4 / 27.6 | 17.2 / 16.9 |
| not measured | not measured | Device A, **native macOS, CGL / Apple M5 GPU** | not measured | not measured | not measured |

The native Mac row is empty on purpose. This work package was not allowed to install pip packages
or run Python on the host. To fill it in, run the macOS command above once (under 60 s) and paste
the printed table here with a new run id.

In the live stack, the GL pass costs more than these isolated numbers suggest. Runs
`plive-20261117-2` and `plive-20261117-3` (2026-11-17, Device A) ran `sim_runner`, its passive
viewer and `perception_node` together in one 2-CPU container. The 160x120 RGB+D pass then averaged
62.3 ms and 87.2 ms, from the node's own log. That
is why `perception_node` renders at the 2 Hz image rate under a CPU budget (`render_budget_frac`),
and why it reuses the last depth map between GL passes while neither the camera nor the arm, legs or crops have moved.

## What a faster renderer would buy

At 160x120 and 2 Hz, the camera costs about 15 % of one core on llvmpipe. In the same run,
`/nandi/game_status` RTF stayed at 1.00 with the camera on and off, so the camera is not the
bottleneck today. The Apple GPU starts to matter in two cases:

- larger images: 640x480 for a real vision model;
- several cameras, or rendering for RL observations.

In those cases the host must run the renderer or the model. A native render or inference server
that talks to the container over a socket would be the next step. It does not exist yet.

The CI gate for this repository stays on the CI runners. This lane is a local measurement
tool, not a CI dependency.
