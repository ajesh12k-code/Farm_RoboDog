# Atharv Nandi — Agricultural Hexapod Mobile Manipulator

A modular **ROS 2 Humble + MuJoCo + Gymnasium + Stable-Baselines3** simulation and control workspace for **Atharv Nandi**, a 25-DOF 6-legged agricultural robot engineered for uneven farm terrain, low soil compaction, active dorsal-arm balance stabilization, and autonomous crop harvesting.

Everything runs in one Docker container: physics, the ROS 2 nodes, a browser control UI and a virtual desktop (noVNC) showing the MuJoCo viewer and the Tk operator console. No local ROS install is needed.

---

## Robot Architecture (Atharv Nandi)

* **18-DOF Hexapod Locomotion (`FR, FL, MR, ML, RR, RL`):** Six 3-DOF legs (`hip_roll`, `thigh_pitch`, `calf_pitch`) capable of switching live between an alternating tripod gait and a trained 23-DOF PPO locomotion/balance policy.
* **Full-Length Dorsal Cargo Basket (`dorsal_basket`):** Top-mounted `0.64 m × 0.24 m` harvest bin seated cleanly above the chassis so all 6 hip joints operate with zero mechanical interference.
* **1-DOF Dorsal Sliding Gantry (`gantry_slide`):** Overhead linear rail (`±0.26 m` travel) spanning the spine above the basket, enabling the manipulator to reach crops ahead of the chassis and slide backward to deposit produce into the bin.
* **6-DOF Ground-Reaching Harvesting Arm & Gripper:** Articulated 4-DOF arm (`arm_yaw`, `arm_shoulder`, `arm_elbow`, `arm_wrist_pitch` with `0.34 m + 0.34 m` links) plus a 2-finger parallel gripper (`finger_left`, `finger_right`; 9 cm open, 4 cm stroke per finger, so it can pinch the 7 cm crops). When empty during locomotion, the 5-DOF dorsal gantry/arm stays tucked at rest and deploys dynamically as a counterweight or ground-bracing stabilizer during steep climbs, slips, or external pushes.

---

## Quick Start (Docker)

Requirements: Docker (Linux, macOS or Windows/WSL 2). From the repository root:

```bash
docker/run.sh
```

The first run builds the `atharv-nandi-sim` image (ROS 2 Humble, MuJoCo 3.3, NumPy 1.26), then starts the whole stack with the repository bind-mounted at `/ws`, so code edits only need a container restart, not a rebuild.

| What | URL |
| --- | --- |
| Browser control UI (keyboard driving, status, scoring) | <http://localhost:8080> |
| Virtual desktop: MuJoCo 3D viewer + Tk operator console | <http://localhost:6080/vnc.html?autoconnect=1&resize=scale> |

`docker/run.sh shell` starts the virtual desktop and drops you into a bash shell (ROS sourced) instead of launching the stack, which is handy for running a single node or `ros2 topic echo`. Stop everything with `Ctrl+C` in the terminal that runs `docker/run.sh` (or close the MuJoCo viewer); closing the Tk console leaves the sim and the browser UI running. Both ports are published on `127.0.0.1` only, because the noVNC desktop has no password.

Rendering is software OpenGL (Mesa llvmpipe), so the real-time factor depends on your CPU; the browser UI and `/nandi/game_status` report it as `rtf`.

### World Configuration (`NANDI_*`)

The world is built once per process by `sim/controllers/farm_scene.py` from these environment variables (every node in the container sees the same values, so all of them agree on the model layout):

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `NANDI_CROPS` | integer ≥ 1 | `6` | Number of crops (`crop_item_1` … `crop_item_N`), scattered in two loose rows ahead of the robot |
| `NANDI_TERRAIN` | `flat`, `ridges`, `rough` | `rough` | `flat` removes the two soil ridges; `rough` adds the ridges plus a patch of low clods further out |
| `NANDI_SEED` | integer | `7` | Seed for the crop scatter and clod placement |

`docker/run.sh` and `docker/run_rl.sh` pass them into the container when they are set, for example:

```bash
NANDI_CROPS=10 NANDI_TERRAIN=flat docker/run.sh
```

The web console listens on `NANDI_WEB_PORT` (default `8080`) inside the container.

---

## Driving the Robot

### Keyboard (browser UI <http://localhost:8080> and Tk console)

| Key | Action |
| --- | --- |
| `W` / `↑` | Walk forward |
| `S` / `↓` | Walk backward |
| `A` / `D` | Strafe left / strafe right |
| `Q` / `E` (`←` / `→` in the browser) | Turn left / turn right (in place) |
| `Space` | Stop: halt the gait now (the Tk console also releases the arm) |
| `H` | Auto-harvest the nearest crop in reach |
| `R` | Reset world (robot + crops back home, score cleared) |
| `G` | Switch gait (Tripod ↔ RL Balance Policy, if a policy is loaded) |
| `Esc` | Release the arm back to RL balance / rest (aborts a running harvest) |
| `V` / `F` | Browser only: show/hide the work envelope / camera follows the robot |

Holding a movement key walks continuously; a quick tap takes one gait cycle; releasing a key finishes the current gait cycle and stops. Keys can be combined (e.g. `W` + `Q` walks forward while turning left). The arrow keys are browser-only; in the Tk console click the panel first so it has keyboard focus. Held keys are dropped when the browser tab or the Tk window loses focus, and as a last safety net locomotion releases a hold by itself if no `cmd_vel` arrives for 0.5 s.

### Tk Operator Console (noVNC desktop)

The console in the virtual desktop drives the same ROS interface: a status panel (position, score, falls, timer, RTF, arm mode/phase), a tap/hold D-pad (tap = one step, hold = walk), **STOP**, **Release Arm**, **Reset World**, **Switch Gait** with the four gait-parameter sliders, **Auto-Harvest**, **Compute Envelope** / **Show/Hide** for the 3D reachability wireframe, and manual sliders for the gantry, arm joints and gripper. Moving a manual slider takes the arm out of balance mode until the arm is released (**Release Arm**, `Esc`, or `/nandi/release_arm`). The browser UI offers the same controls plus a live Three.js view of the world.

### Locomotion Modes

* **Tripod Gait** (default, always available): deterministic alternating tripod gait in `sim/controllers/tripod_gait.py`, tuned by the gait parameters below.
* **RL Balance Policy**: 23-DOF PPO policy that drives the 18 leg joints and uses the dorsal arm as a counterweight. It needs a trained policy archive in `rl_training/models/` **and** the training image (`docker/run_rl.sh`), because the default runtime image does not include PyTorch. Without a policy the toggle answers `No RL policy loaded - using Tripod gait` and the robot stays on the tripod gait; `/nandi/locomotion_status` always reports the truth (`mode`, `policy_loaded`, `policy_path`).

Gait parameters (`/nandi/gait_params`):

| Index | Parameter | Range | Default |
| --- | --- | --- | --- |
| 0 | `speed_scale` | 0.2 – 1.5 | 1.0 |
| 1 | `step_height_scale` | 0.5 – 2.0 | 1.0 |
| 2 | `freq_hz` | 0.8 – 3.0 | 1.8 |
| 3 | `balance_gain` | 0 – 2 | 1.0 |

### Harvesting

Walk until a crop is within **1.08 m** (horizontal distance to the trunk) and press **Auto-Harvest** (`H`). The UI stops the gait first; the harvest node then takes the arm out of balance mode, picks the nearest crop that is not already in the basket or under the robot's body (closed-form IK refined in the world frame, gripper pointing straight down), descends, closes the gripper, lifts it, checks the grasp (the crop must have risen at least 5 cm and sit within 6 cm of the gripper tip; one regrasp is attempted on failure), swings it over the left side of the basket, releases it, verifies it landed inside the basket footprint (the same rule the score uses), and always returns the arm to rest. Progress is published on `/nandi/harvest_status`.

* If nothing is within 1.08 m it reports the nearest crop and its distance, e.g. `No crop in reach (nearest crop_item_3 1.42 m)`, or `No crop in reach (all crops already in basket)`. A crop under the body is never attempted (the arm would have to pass through the trunk); if it is the nearest, the message ends in `under the body - back up`.
* The arm itself reaches about 0.86 m ahead of the trunk and about 0.55 m to the side. A crop inside 1.08 m but beyond that is reported as `crop_item_k out of arm reach (X.XX m), move closer` and the arm goes back to rest; walk closer and press `H` again.
* **Release Arm** (`Esc`), the Tk console's **STOP**, a manual arm slider move or **Reset World** aborts a running harvest.

### Scoring Game

`sim_runner` keeps score and publishes it on `/nandi/game_status` (shown in the browser UI):

* **In basket:** a crop counts once its centre is inside the basket footprint in the trunk frame (`|x| < 0.31 m`, `|y| < 0.11 m`, trunk-frame `z` between the basket floor top, 0.061 m, and +0.12 m, just under the wall tops). The goal is to get all `crops_total` crops into the basket.
* **Falls:** tilt above 50° or trunk height below 0.12 m counts as a fall (once per fall).
* **Timer:** `elapsed_s` since the last reset; `rtf` is the measured real-time factor.

**Reset World** (`R`, `/nandi/reset_world`) puts the robot and every crop back to the home keyframe, zeroes the falls counter and the timer, clears the work-envelope wireframe, and asks locomotion to stop and the harvest node to release the arm.

---

## ROS 2 Interface

All JSON payloads are `std_msgs/String`. Joint vectors are `std_msgs/Float64MultiArray` in the order given by the model (arm: gantry, yaw, shoulder, elbow, wrist, finger_l, finger_r; legs: FR, FL, MR, ML, RR, RL × hip, thigh, calf).

### Topics

| Topic | Type | Publisher → Subscriber | Payload |
| --- | --- | --- | --- |
| `/nandi/cmd_vel` | `geometry_msgs/Twist` | UIs → locomotion | `linear.x` = vx ∈ [-1, 1] (forward +), `linear.y` = vy ∈ [-1, 1] (left +), `angular.z` = yaw ∈ [-1, 1] (+ = turn left / CCW), `linear.z` = `1.0` hold (keep walking while commands keep arriving) or `0.0` single step (one gait cycle). All zero with `linear.z = 0` = release (finish the cycle, stop). Senders re-publish a held command at ≥ 5 Hz; a hold with no message for 0.5 s is treated as released. |
| `/nandi/gait_params` | `Float64MultiArray` | UIs → locomotion | `[speed_scale, step_height_scale, freq_hz, balance_gain]`, clamped to the ranges above |
| `/nandi/locomotion_status` | `String` (JSON, 2 Hz) | locomotion → UIs | `{"mode": "tripod"\|"rl", "policy_loaded": bool, "policy_path": str\|null, "gait_params": [4], "walking": bool}` |
| `/nandi/leg_targets` | `Float64MultiArray` (18) | locomotion → sim_runner | Leg joint position targets |
| `/nandi/rl_arm_targets` | `Float64MultiArray` (5) | locomotion → harvest | Arm balance targets from the RL policy (gantry + 4 arm joints) |
| `/nandi/manual_arm` | `Float64MultiArray` (7) | UIs → harvest | Manual arm + gripper targets (switches the arm to manual mode) |
| `/nandi/arm_targets` | `Float64MultiArray` (7) | harvest → sim_runner | Final arm + gripper targets after arbitration |
| `/nandi/harvest_status` | `String` (JSON, 2 Hz + on change) | harvest → UIs | `{"mode": "rl_balance"\|"ik_harvest"\|"manual", "phase": str, "target": str\|null, "last_result": str}` |
| `/nandi/sim_qpos` | `Float64MultiArray` (nq) | sim_runner → all | Full world `qpos`: trunk `[0:7]`, arm `[7:14]`, legs `[14:32]`, then 7 per crop |
| `/nandi/game_status` | `String` (JSON, 2 Hz) | sim_runner → UIs | `{"crops_total": int, "in_basket": int, "in_basket_names": [str], "falls": int, "fallen": bool, "elapsed_s": float, "rtf": float}` |
| `/nandi/workspace_wireframe` | `String` (JSON) | UIs, sim_runner (on reset) → sim_runner, UIs | Reachability wireframe segments to draw, trunk frame (empty string clears) |

### Services (all `std_srvs/Trigger`)

| Service | Server | Behaviour |
| --- | --- | --- |
| `/nandi/toggle_locomotion_mode` | locomotion | Toggles Tripod ↔ RL; replies `Mode: RL Balance Policy` / `Mode: Tripod Gait`, or `success=false`, `No RL policy loaded - using Tripod gait` when no policy is loaded |
| `/nandi/stop_motion` | locomotion | Stops the gait immediately |
| `/nandi/auto_harvest` | harvest | Starts the asynchronous pick-and-place of the nearest crop in reach (see Harvesting) |
| `/nandi/release_arm` | harvest | Leaves manual / IK mode for RL balance, aborting a running harvest; the arm returns to rest |
| `/nandi/reset_world` | sim_runner | Resets robot and crops to the home keyframe, zeroes falls and timer, clears the wireframe, then calls `stop_motion` and `release_arm` |
| `/compute_work_envelope` | work envelope | Samples the arm's joint limits and returns the reachability wireframe JSON in the message |

### Nodes

| Node | Role |
| --- | --- |
| `sim/sim_runner.py` | Steps MuJoCo physics, hosts the passive 3D viewer, draws the body-attached work-envelope wireframe, publishes `sim_qpos` and `game_status`, serves `reset_world` |
| `sim/locomotion_service.py` | Turns `cmd_vel` + `gait_params` into leg targets via `TripodGaitGenerator` or `RLBalanceController`; serves mode toggle and stop |
| `sim/harvest_ik_service.py` | Arbitrates arm ownership (RL balance / IK harvest / manual) and runs the DLS IK pick-and-place sequence |
| `sim/work_envelope_service.py` | Introspects the MJCF manipulator joints and limits to compute a 3D reachability wireframe on demand |
| `sim/control_panel_gui.py` | Tk operator console, built on `NandiROSClient` (`controllers/ros_client.py`) |
| `sim/web_bridge.py` | Serves the browser UI (`web/`) and a WebSocket on `:8080`; streams world poses + status at 30 Hz and forwards browser commands to the topics and services above |

All kinematics, gait math, RL inference, scoring and client logic live in pure Python modules under `sim/controllers/`, so the ROS nodes stay thin and the logic is unit-testable without ROS. Every live node builds the same world through `controllers.farm_scene.load_farm_model()`, which keeps `nq` and the `qpos` layout identical across processes.

---

## Tests

Unit tests in `tests/` exercise the pure modules in `sim/controllers/` and never import ROS. Run them in the sim image:

```bash
docker build -t atharv-nandi-sim -f docker/Dockerfile .
docker run --rm -v "$PWD:/ws" -w /ws atharv-nandi-sim python3 -m pytest -q
```

or directly on any Python 3.10+ machine without ROS (CI uses Python 3.11):

```bash
pip install -r requirements-ci.txt
python -m pytest -q
```

---

## Training the RL Balance Policy

The trained policy archive is **not** included in the repository (`rl_training/models/*.zip` is git-ignored). Train it with:

```bash
docker/train.sh
```

This builds `atharv-nandi-train` (the sim image plus CPU PyTorch, Stable-Baselines3 and Gymnasium) and runs `rl_training/train_nandi_balance.py` headless with the repository mounted. The trainer runs 4 parallel MuJoCo workers for 1.2 M timesteps, saves a checkpoint every 100 k total steps as `rl_training/models/nandi_balance_<steps>_steps.zip`, resumes automatically from the newest checkpoint, and writes `rl_training/models/nandi_balance_final.zip` at the end.

> **Training is heavy.** Expect hours of sustained load on all cores. Run it on a Linux workstation, a GPU box or a cloud VM rather than a laptop, then copy the resulting `.zip` into `rl_training/models/` on the machine that runs the sim. It runs fine unattended (`nohup docker/train.sh > train.log 2>&1 &`).

The training environment loads the raw robot MJCF (`sim/models/AtharvNandi/atharv_nandi.xml`), and the runtime observation builder only reads the robot part of `qpos` (`[0:32]`), so a policy trained here works unchanged in the multi-crop farm world.

`docker/train.sh <command>` runs any other command in `rl_training/` with the training image, e.g. `docker/train.sh python3 train_nandi.py` for the 18-DOF leg-only policy.

### Running the Sim with the Policy

```bash
docker/run_rl.sh
```

Same UI and ports as `docker/run.sh`, but on the training image so the locomotion node can load the policy (it prefers `nandi_balance_final.zip`, else the newest checkpoint). Press **Switch Gait** (`G`) to enable RL mode. To watch the push-recovery evaluation viewer, run `docker/run_rl.sh shell`, open the noVNC desktop and start `cd /ws/rl_training && python3 evaluate_nandi_balance.py`.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt                         # Full Python stack for a native (non-Docker) setup
├── requirements-ci.txt                      # Minimal deps for the ROS-free unit tests
├── pytest.ini
├── docker/
│   ├── Dockerfile                           # atharv-nandi-sim: ROS 2 Humble + MuJoCo + Xvfb/noVNC
│   ├── Dockerfile.train                     # atharv-nandi-train: sim image + CPU torch + SB3 + Gymnasium
│   ├── run.sh                               # Build + run the sim (UI :8080, noVNC :6080)
│   ├── run_rl.sh                            # Same, on the training image (RL mode available)
│   ├── train.sh                             # Headless PPO training
│   └── start.sh                             # Container entry: virtual desktop, then the launcher
├── sim/
│   ├── models/
│   │   ├── AtharvNandi/
│   │   │   └── atharv_nandi.xml  # 25-DOF hexapod + dorsal gantry/arm MJCF
│   │   └── unitree_a1/                      # Baseline quadruped MJCF reference
│   ├── controllers/                         # Pure Python logic decoupled from ROS 2 nodes
│   │   ├── farm_scene.py                    # Shared world builder (robot + crops + terrain)
│   │   ├── game_status.py                   # Scoring, basket footprint, fall detection, RTF
│   │   ├── locomotion_logic.py              # cmd_vel parsing, hold dead-man, gait params, mode
│   │   ├── tripod_gait.py                   # 18-DOF alternating tripod gait (walk, strafe, turn, balance)
│   │   ├── rl_balance_policy.py             # 23-DOF PPO inference & 84-D obs builder
│   │   ├── arm_ik_solver.py                 # Arm IK solver + work-envelope computation
│   │   ├── harvest_logic.py                 # Target choice, grasp/drop checks, pick-and-place sequence
│   │   ├── key_drive.py                     # Keyboard / D-pad -> cmd_vel logic for the Tk console
│   │   ├── ros_client.py                    # ROS 2 service & topic client used by the console
│   │   ├── telemetry.py                     # Telemetry + status parsing / formatting
│   │   ├── gui_widgets.py                   # Tk widget helpers
│   │   └── web_scene.py                     # Browser protocol: scene/pose serialisation, validation
│   ├── sim_runner.py                        # MuJoCo physics host, viewer, scoring
│   ├── locomotion_service.py                # Thin ROS 2 node (Tripod gait ↔ RL policy)
│   ├── harvest_ik_service.py                # Thin ROS 2 node (RL balance ↔ IK harvest ↔ manual)
│   ├── work_envelope_service.py             # 3D workspace wireframe service
│   ├── control_panel_gui.py                 # Tk operator console
│   ├── web_bridge.py                        # Browser UI server + ROS 2 WebSocket bridge (:8080)
│   └── launch_nandi.sh                   # Starts all nodes + console (used by docker/start.sh)
├── web/                                     # Browser UI (Three.js, loaded from the jsDelivr CDN)
├── tests/                                   # ROS-free pytest suite
└── rl_training/
    ├── models/                              # Trained policies go here (git-ignored)
    ├── nandi_balance_env.py             # 23-DOF whole-body Gymnasium environment
    ├── disturbance_manager.py               # Uneven farm ridges + lateral push disturbances
    ├── balance_reward.py                    # Rest-pose enforcement & arm-support reward
    ├── train_nandi_balance.py           # 4-worker PPO trainer (auto-resume)
    ├── evaluate_nandi_balance.py        # Camera-tracking push-recovery viewer
    ├── nandi_env.py                     # 18-DOF leg-only Gymnasium environment
    ├── terrain_manager.py                   # 18-DOF leg-only uneven ridge randomizer
    ├── reward_functions.py                  # 18-DOF locomotion reward & fall check
    ├── train_nandi.py                   # 18-DOF PPO trainer
    └── evaluate_nandi.py                # 18-DOF evaluation viewer
```

---

## Native Setup (without Docker)

On Ubuntu 22.04 with ROS 2 Humble installed (`ros-humble-ros-base` plus `geometry_msgs`, `std_msgs`, `std_srvs`, `sensor_msgs`):

```bash
sudo apt-get install -y python3-tk
pip3 install "numpy>=1.24,<2" "mujoco>=3.3,<3.4" "websockets>=13.1,<14"
# optional, for RL mode and training: pip3 install torch stable-baselines3 gymnasium
source /opt/ros/humble/setup.bash
sim/launch_nandi.sh
```

Keep NumPy below 2: ROS 2 Humble's Python bindings are built against NumPy 1.x.

The web console has no login, so natively it listens on `127.0.0.1` only. To reach it from another machine set `NANDI_WEB_HOST=0.0.0.0` and list the hostname you browse to in `NANDI_WEB_ALLOWED_HOSTS` (comma-separated); browsers using any other hostname are refused, which also blocks DNS-rebinding pages. Docker already sets `NANDI_WEB_HOST=0.0.0.0` inside the container and publishes the port on `127.0.0.1`.

## License

Apache License 2.0, see [LICENSE](LICENSE). The bundled Unitree A1 reference model in `sim/models/unitree_a1/` keeps its own license.
