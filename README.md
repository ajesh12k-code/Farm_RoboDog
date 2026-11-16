# FarmDog6 — Agricultural Hexapod Mobile Manipulator

A modular **ROS 2 Humble + MuJoCo + Gymnasium + Stable-Baselines3** simulation and control workspace for **KisaanuNexusFarmDog6**, a 25-DOF 6-legged agricultural robot engineered for uneven farm terrain, low soil compaction, active dorsal-arm balance stabilization, and autonomous crop harvesting.

---

## Robot Architecture (`KisaanuNexusFarmDog6`)

* **18-DOF Hexapod Locomotion (`FR, FL, MR, ML, RR, RL`):** Six 3-DOF legs (`hip_roll`, `thigh_pitch`, `calf_pitch`) capable of switching live between an alternating tripod gait and a trained 23-DOF PPO locomotion/balance policy.
* **Full-Length Dorsal Cargo Basket (`dorsal_basket`):** Top-mounted `0.64 m × 0.24 m` harvest bin seated cleanly above the chassis so all 6 hip joints operate with zero mechanical interference.
* **1-DOF Dorsal Sliding Gantry (`gantry_slide`):** Overhead linear rail (`±0.26 m` travel) spanning the spine above the basket, enabling the manipulator to reach crops ahead of the chassis and slide backward to deposit produce into the bin.
* **6-DOF Ground-Reaching Harvesting Arm & Gripper:** Articulated 4-DOF arm (`arm_yaw`, `arm_shoulder`, `arm_elbow`, `arm_wrist_pitch` with `0.34 m + 0.34 m` links) plus a 2-finger parallel gripper (`finger_left`, `finger_right`). When empty during locomotion, the 5-DOF dorsal gantry/arm stays tucked at rest and deploys dynamically as a counterweight or ground-bracing stabilizer during steep climbs, slips, or external pushes.

---

## Repository Structure

```text
ros2_work_space/
├── requirements.txt                         # Python dependencies (MuJoCo, Gymnasium, SB3)
├── README.md                                # System overview, architecture, and usage guide
├── sim/
│   ├── models/
│   │   ├── KisaanuNexusFarmDog6/
│   │   │   └── kisaanu_nexus_farm_dog6.xml  # 25-DOF hexapod + dorsal gantry/arm MJCF
│   │   └── unitree_a1/                      # Baseline quadruped MJCF reference
│   ├── controllers/                         # Pure Python logic decoupled from ROS 2 nodes
│   │   ├── tripod_gait.py                   # 18-DOF alternating tripod gait generator
│   │   ├── rl_balance_policy.py             # 23-DOF PPO inference & 84-D obs builder
│   │   ├── arm_ik_solver.py                 # Damped Least-Squares Jacobian IK solver
│   │   └── ros_client.py                    # Async ROS 2 service & topic client bridge
│   ├── work_envelope_service.py             # Model-agnostic 3D workspace ROS 2 service
│   ├── locomotion_service.py                # Thin ROS 2 node (RL Policy ↔ Tripod Gait)
│   ├── harvest_ik_service.py                # Thin ROS 2 node (RL Balance ↔ IK Harvest)
│   ├── sim_runner.py                        # MuJoCo physics host & 3D wireframe renderer
│   ├── control_panel_gui.py                 # Compact Tkinter operator console view
│   └── launch_farmdog6.sh                   # One-command launcher for services + GUI
├── src/
│   └── quadruped_controller/                # ROS 2 Humble ament_python controller package
└── rl_training/
    ├── models/
    │   └── farmdog6_balance_final.zip       # Trained 23-DOF PPO balance & walking policy
    ├── disturbance_manager.py               # Uneven farm ridges + lateral push disturbances
    ├── balance_reward.py                    # Rest-pose enforcement & arm-support reward
    ├── farm_dog6_balance_env.py             # 23-DOF whole-body Gymnasium environment
    ├── train_farm_dog6_balance.py           # Thread-safe 4-worker PPO trainer (WSL 2 ready)
    ├── evaluate_farm_dog6_balance.py        # Camera-tracking MuJoCo push-recovery viewer
    ├── terrain_manager.py                   # 18-DOF leg-only uneven ridge randomizer
    ├── reward_functions.py                  # 18-DOF locomotion reward & fall check
    ├── farm_dog6_env.py                     # 18-DOF leg-only Gymnasium environment
    ├── train_farm_dog6.py                   # 18-DOF PPO trainer
    └── evaluate_farm_dog6.py                # 18-DOF camera-tracking evaluation viewer

Modular ROS 2 Service & Controller Architecture

All kinematics, gait math, RL inference, and client networking reside in sim/controllers/, keeping the ROS 2 service nodes compact:

1. 

Node / Module                 

sim_runner.py


Interfaces (Topics & Services)          

Publishes /farmdog6/sim_qpos

Subscribes /farmdog6/leg_targets, /farmdog6/arm_targets, /farmdog6/workspace_wireframe


Functionality

Steps MuJoCo physics at 500 Hz, syncs the 3D viewer, and renders dynamic body-attached workspace wireframes.


2.

Node / Module

locomotion_service.py


Interfaces (Topics & Services)

Services /farmdog6/stop_motion, /farmdog6/toggle_locomotion_mode

Publishes /farmdog6/leg_targets, /farmdog6/rl_arm_targets


Functionality

Delegates to RLBalanceController (23-DOF PPO) or TripodGaitGenerator (18-DOF tripod) and toggles live via ROS 2 Trigger.


3.

Node / Module

harvest_ik_service.py


Interfaces (Topics & Services)

Service /farmdog6/auto_harvest

Subscribes /farmdog6/rl_arm_targets, /farmdog6/manual_arm

Publishes /farmdog6/arm_targets


Functionality

Arbitrates arm ownership: forwards tucked/stabilizing RL arm commands while walking and hands off to DorsalArmIKSolver when harvesting.

4.

Node / Module

work_envelope_service.py



Interfaces (Topics & Services)

Service /compute_work_envelope


Functionality

Dynamically introspects MJCF manipulator joints and limits on demand to compute a translucent 3D reachability wireframe.



5.

Node / Module
control_panel_gui.py


Interfaces (Topics & Services)

Uses FarmDog6ROSClient (controllers/ros_client.py)


Functionality

Compact Tkinter UI providing tap/hold D-pad movement, Switch Gait (RL ↔ Tripod) toggle, Auto-Harvest, workspace envelope controls, and manual joint sliders.


Installation & Setup (Ubuntu 22.04 / WSL 2)

1. Install System & ROS 2 Dependencies:

sudo apt-get update && sudo apt-get install -y python3-tk

2. Install Python Packages:

cd ~/ros2_work_space
pip3 install -r requirements.txt

3. Build ROS 2 Workspace Packages:

cd ~/ros2_work_space
PYTHONNOUSERSITE=1 colcon build
source install/setup.bash

Running KisaanuNexusFarmDog6

1. Launch the Live ROS 2 Simulation & Operator Console

~/ros2_work_space/sim/launch_farmdog6.sh


Locomotion & Gait Toggle: Tap ▲ Forward, ▼ Backward, ◄ Left, or Right ► for a single step, or hold to walk continuously. Click 🤖 Switch Gait (RL ↔ Tripod) anytime to toggle between the trained 23-DOF PPO balance policy and the deterministic 18-DOF tripod gait.

Autonomous Harvesting: Walk within 1.08 m of crop_item_1 and click 🌾 Auto-Harvest (IK Pick & Basket) to switch the dorsal arm from RL balance mode to Jacobian IK pick-and-place, deposit the crop into the top basket, and hand control back to the RL policy.

Dynamic Work Envelope: Click 🔄 Compute Envelope to project the 3D reachability wireframe into the live simulation, and 👁 Show/Hide to toggle visibility.

Train or Evaluate the 23-DOF Arm-Assisted Balance Policy

cd ~/ros2_work_space/rl_training
python3 train_farm_dog6_balance.py
LIBGL_ALWAYS_SOFTWARE=1 python3 evaluate_farm_dog6_balance.py


