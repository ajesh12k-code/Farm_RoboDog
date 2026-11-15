# KisaanuNexusFarmDog6 — Agricultural Hexapod Mobile Manipulator

A modular **ROS 2 Humble + MuJoCo + Gymnasium** simulation and control workspace for **KisaanuNexusFarmDog6**, a 25-DOF 6-legged agricultural robot engineered for uneven farm terrain, low soil compaction, and autonomous crop harvesting.

---

## Robot Architecture (`KisaanuNexusFarmDog6`)

* **18-DOF Hexapod Locomotion (`FR, FL, MR, ML, RR, RL`):** Six 3-DOF legs (`hip_roll`, `thigh_pitch`, `calf_pitch`) using an alternating tripod gait (`3` planted feet at all times) to distribute payload weight evenly across soft soil and maintain static stability on steep or ridged terrain.
* **Full-Length Dorsal Cargo Basket (`dorsal_basket`):** Top-mounted `0.64 m × 0.24 m` harvest bin seated cleanly above the chassis so all 6 hip joints operate with zero mechanical interference.
* **1-DOF Dorsal Sliding Gantry (`gantry_slide`):** Overhead linear rail (`±0.26 m` travel) spanning the spine above the basket, enabling the manipulator to reach crops ahead of the chassis and slide backward to deposit produce into the bin.
* **6-DOF Ground-Reaching Harvesting Arm & Gripper:** Articulated 4-DOF arm (`arm_yaw`, `arm_shoulder`, `arm_elbow`, `arm_wrist_pitch` with `0.34 m + 0.34 m` links) plus a 2-finger parallel gripper (`finger_left`, `finger_right`) capable of reaching floor level (`z ≈ 0.015 m`).

---

## Repository Structure

```text
ros2_work_space/
├── requirements.txt                  # Python dependencies (MuJoCo, Gymnasium, SB3)
├── README.md                         # System overview, architecture, and usage guide
├── sim/
│   ├── models/
│   │   ├── KisaanuNexusFarmDog6/
│   │   │   └── kisaanu_nexus_farm_dog6.xml  # 25-DOF hexapod + dorsal gantry/arm MJCF
│   │   └── unitree_a1/               # Baseline quadruped MJCF reference
│   ├── work_envelope_service.py      # Model-agnostic kinematic 3D workspace ROS 2 service
│   ├── locomotion_service.py         # 18-DOF tripod gait generator ROS 2 node/service
│   ├── harvest_ik_service.py         # Jacobian DLS IK & 6-stage pick-and-place service
│   ├── sim_runner.py                 # MuJoCo physics host, state publisher, & wireframe renderer
│   ├── control_panel_gui.py          # Thin Tkinter operator console (services + overrides)
│   ├── launch_farmdog6.sh            # One-command launcher for all ROS 2 services + GUI
│   └── mujoco_bridge.py              # Baseline ROS 2 joint bridge
├── src/
│   └── quadruped_controller/         # ROS 2 Humble ament_python controller package
└── rl_training/
    ├── a1_env.py                     # Headless Gymnasium RL environment
    └── train_ppo.py                  # Multi-core CPU PPO training script
Modular ROS 2 Service ArchitectureThe simulation is decoupled into independent ROS 2 nodes and services so kinematics, gait generation, inverse kinematics, and visualization remain modular and model-agnostic:Node / ScriptInterfaces (Topics & Services)Functionalitysim_runner.pyPublishes /farmdog6/sim_qposSubscribes /farmdog6/leg_targets, /farmdog6/arm_targets, /farmdog6/workspace_wireframeSteps MuJoCo physics at 500 Hz, syncs the 3D viewer, and renders dynamic body-attached wireframe markers.
locomotion_service.py	

Service /farmdog6/stop_motion

Subscribes /farmdog6/cmd_vel

Publishes /farmdog6/leg_targets
	Generates the 18-DOF alternating tripod gait (supports single-step tap or continuous hold walking).
harvest_ik_service.py	

Service /farmdog6/auto_harvest

Subscribes /farmdog6/sim_qpos, /farmdog6/manual_arm

Publishes /farmdog6/arm_targets
	Runs a Damped Least-Squares Jacobian IK solver (mj_jacSite) and 6-stage state machine (hover → descend → grasp → lift → slide over basket → release).
work_envelope_service.py	Service /compute_work_envelope	Dynamically introspects MJCF manipulator joints and limits on demand to compute a continuous translucent 3D reachability wireframe.
control_panel_gui.py	Service & topic client	Lightweight Tkinter operator UI providing tap/hold D-pad movement, Auto-Harvest trigger, Compute Envelope button, and manual joint sliders.
work_envelope_service.py	Service /compute_work_envelope	Dynamically introspects MJCF manipulator joints and limits on demand to compute a continuous translucent 3D reachability wireframe.
control_panel_gui.py	Service & topic client	Lightweight Tkinter operator UI providing tap/hold D-pad movement, Auto-Harvest trigger, Compute Envelope button, and manual joint sliders.
Build ROS 2 Workspace Packages:
    Bash

    cd ~/ros2_work_space
    PYTHONNOUSERSITE=1 colcon build
    source install/setup.bash

Running KisaanuNexusFarmDog6

Launch all four ROS 2 services (work_envelope_service, locomotion_service, harvest_ik_service, sim_runner) and the operator control panel with a single command:
Bash

~/ros2_work_space/sim/launch_farmdog6.sh

    Locomotion: Tap ▲ Forward, ▼ Backward, ◄ Left, or Right ► once for a single tripod stride, or hold to walk continuously.

    Autonomous Harvesting: Click 🌾 Auto-Harvest (IK Pick & Basket) to solve IK for the target crop on the ground, grasp it, slide along the dorsal gantry over the top basket, and deposit it inside.

    Dynamic Work Envelope: Click 🔄 Compute Envelope to query /compute_work_envelope and project the translucent 3D reachability wireframe directly into the live simulation.
    EOF

