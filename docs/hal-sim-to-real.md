# Sim-to-real: hardware abstraction and estimation

Status: design only, no hardware exists (rover hardware is out of scope for now). Gaps covered: AUT-02 (HAL), AUT-05 (state estimation), AUT-13 (noise and latency), plus the Mac GPU question.

## 1 The problem with today's interface

Every built node reads the full MuJoCo state from one topic, `/nandi/sim_qpos` (`Float64MultiArray`, trunk `[0:7]`, arm `[7:14]`, legs `[14:32]`, then 7 per crop; README, ROS 2 Interface). A real robot has no such topic: it has joint encoders, an IMU, maybe a GNSS receiver and a camera, and it has to estimate the trunk pose and find crops. Each consumer of `sim_qpos` is therefore a sim-to-real dependency.

| Consumer (code) | What it takes from `sim_qpos` | Real replacement |
|---|---|---|
| `locomotion_service.py` (tripod balance, RL observation) | trunk orientation, joint positions, velocities by difference | `/nandi/imu` + `/nandi/joint_states` |
| `harvest_ik_service.py` / `harvest_logic.py` | trunk pose, arm joints, crop positions, world-reset detection | `/nandi/joint_states`, `/nandi/pose_estimate`, `/nandi/crop_detections` |
| `web_bridge.py`, `control_panel_gui.py` | all poses for drawing | `/nandi/pose_estimate`, `/nandi/joint_states`, `/nandi/crop_detections` |
| `sim_runner.py` scoring (`game_status`) | trunk and crop poses for basket truth and falls | basket truth has no real equivalent; use a basket load cell or a top camera [ASSUMED design]; falls from IMU tilt |

## 2 Target interface

Use standard ROS 2 message types so that drivers, `ros2_control` and recorders work unchanged. All names are [DESIGNED] in this document; none is in code.

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/nandi/joint_states` | `sensor_msgs/JointState` | 100 Hz [ASSUMED] | 25 joints by name (the MJCF joint names), position, velocity, effort |
| `/nandi/imu` | `sensor_msgs/Imu` | 200 Hz [ASSUMED] | trunk frame, with covariances |
| `/nandi/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | 50 Hz [ASSUMED] | world (field) frame; covariance is mandatory, consumers gate on it |
| `/nandi/joint_commands` | `trajectory_msgs/JointTrajectory` or `std_msgs/Float64MultiArray` per controller | 50 Hz [ASSUMED] | replaces `leg_targets` + `arm_targets` at the HAL boundary |
| `/nandi/crop_detections` | see [hld.md](hld.md) section 3.1 | per frame | must carry covariance (`cov`) and `data_provenance` |

Two ways to implement the HAL:

| Option | What it is | For | Against |
|---|---|---|---|
| A `ros2_control` | hardware interface plugin per bus (sim plugin `mujoco_ros2_control`-style, real plugin per servo bus), `joint_trajectory_controller`, `joint_state_broadcaster` | standard, real-time loop, many drivers | C++ plugins, a heavier build; the existing nodes are Python |
| B Thin Python HAL | one `hal_node` with a `SimBackend` (MuJoCo, today's `sim_runner`) and a `RealBackend` (servo bus driver), both publishing the topics above | small, fits the current Python layout, unit-testable without ROS like `atharv_nandi/core/` | non-standard; real-time behaviour depends on Python timing |

Proposal: B for the sim pilot and the first bench leg (fast, testable), with the topic names and types of A so that a later move to `ros2_control` does not change any consumer. This is a proposal, not a decision; it depends on the open software-stack question.

## 3 Migration order from `/nandi/sim_qpos`

Each step keeps the full test suite green and keeps `sim_qpos` published until the last consumer has moved.

| Step | Change | Consumers moved | Check |
|---|---|---|---|
| M1 | `sim_runner` also publishes `/nandi/joint_states` (the safety package already adds this [PROPOSED]) and `/nandi/imu` from MuJoCo (exact values) | none | a unit test compares them with `qpos` |
| M2 | `sim_runner` publishes `/nandi/pose_estimate` = truth + zero covariance | none | same |
| M3 | locomotion reads `joint_states` + `imu` | locomotion | fall targets P-05/P-06 unchanged within noise |
| M4 | perception package: `/nandi/crop_detections` from truth + noise [PROPOSED]; harvest target selection uses detections | harvest target choice | P-01 re-measured |
| M5 | harvest IK reads `joint_states` + `pose_estimate`; grasp hold check moves from crop truth to finger position error and effort (the false-positive lesson, [state-machines.md](state-machines.md) 3.1) | harvest | P-01, P-03 re-measured |
| M6 | UIs read the new topics | web_bridge, Tk console | manual check |
| M7 | `sim_qpos` kept only for scoring and the recorder (truth for KPIs), renamed `/nandi/sim_truth` [ASSUMED name] so no control node can subscribe by accident | scoring only | grep: no control node under `sim/` subscribes to it |

## 4 Noise and latency layer

A `SimBackend` option, off by default, that degrades the M1/M2 topics before publishing. Values are starting points [ASSUMED] to be replaced by datasheet or bench values.

| Signal | Model | Starting value |
|---|---|---|
| Joint position | Gaussian noise + quantisation + backlash dead-band | 0.002 rad, 4096 counts per turn, 0.01 rad dead-band [ASSUMED] |
| Joint command | transport delay + first-order lag | 10 ms delay, 30 ms time constant [ASSUMED] |
| IMU | white noise + bias random walk | 0.01 rad/s gyro noise, slow bias [ASSUMED] |
| Pose estimate | drift proportional to distance + covariance growth; resets at a "fix" | 2 % of distance [ASSUMED] |
| Detections | position noise, 10 % dropout, 100 ms latency [ASSUMED], false positives on clods | perception package [PROPOSED] |
| Comms | message drop and delay on `cmd_vel` | for HZ-06 tests |

Rule: every KPI run records whether the layer was on and with which values (run record). A pass without noise is labelled "ideal sensors".

## 5 Mac GPU finding

Question: can the Mac GPU be used too?

| Fact | Evidence |
|---|---|
| The development Mac has an Apple M5 GPU with Metal 4 support | `system_profiler SPDisplaysDataType` on 2026-11-17 [MEASURED, laptop, read-only query] |
| The sim container runs in a Linux VM (Ubuntu, aarch64) and sees no GPU device (`/dev/dri` and `/dev/nvidia*` empty) | `docker info` and run D-02 on 2026-11-17 [MEASURED] |
| Docker on macOS does not pass Metal through to Linux containers; inside the container, rendering is Mesa llvmpipe on the CPU | README (Quick Start: "Rendering is software OpenGL (Mesa llvmpipe)") and the empty device list above; the general Docker-on-macOS limitation is [UNVERIFIED] against vendor docs in this work |
| MuJoCo physics is CPU-bound and fast already: 45 to 49 x real time on 2 container CPUs with no rendering | run D-02 [MEASURED] |

So, yes, but only natively, and only for two things:

1. **Rendering.** The MuJoCo viewer or the browser console run natively on macOS use the GPU. The browser console at <http://localhost:8080> already renders with the Mac's GPU, because Three.js runs in the browser on the host, not in the container. The noVNC desktop does not (it is rendered by llvmpipe in the container).
2. **Inference.** A detector or policy can run natively with Core ML or PyTorch's Metal backend for experiments. This does not represent the edge target (Jetson or Hailo speed is not provable on a Mac, cannot-prove list), and in-house training on the Mac is a non-goal.

It does not help physics (CPU), headless evaluation, or CI. The project rule is that checks run on the CI runners, not on the laptop. Native setup and measurement notes belong in `tools/mac_gpu/README.md` [PROPOSED, being written by another work package; not on this branch].
