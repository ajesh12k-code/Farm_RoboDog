# Quadruped Robot Locomotion & Control (ROS 2 + MuJoCo)

A CPU-first robotics development and reinforcement learning (RL) prototyping environment for designing, simulating, and controlling a 12-DOF quadruped robot (Unitree A1) using **ROS 2 Humble** and **MuJoCo**.

## Project Objectives

* **Decoupled Simulation & Middleware:** Separate low-level rigid-body contact physics (MuJoCo) from high-level robot state estimation, gait generation, and motor command routing (ROS 2 Humble).
* **CPU-First Local Prototyping:** Enable fast, deterministic physics simulation and controller debugging on standard x64 hardware without requiring an NVIDIA CUDA GPU locally, while maintaining a clean architecture ready for cloud GPU RL policy training (PyTorch / Stable Baselines3).
* **Sim-to-Real Ready Architecture:** Structure control nodes using standard ROS 2 message interfaces (`sensor_msgs`, `geometry_msgs`, `std_msgs`) so policies and controllers tested in simulation can transfer directly to physical quadruped hardware.

---

## System Architecture

```text
+---------------------------------------------------------------+
| Windows Host OS                                               |
|  +---------------------------------------------------------+  |
|  | WSL 2 Utility VM (Ubuntu 22.04 Jammy)                   |  |
|  |                                                         |  |
|  |  +------------------------+   +----------------------+  |  |
|  |  | ROS 2 Humble Workspace |   | MuJoCo Physics       |  |  |
|  |  | (quadruped_controller) |<->| (Unitree A1 Model)   |  |  |
|  |  +------------------------+   +----------+-----------+  |  |
|  +------------------------------------------|--------------+  |
|                                             v (WSLg)          |
|                                  [Interactive 3D Viewer]      |
+---------------------------------------------------------------+

Prerequisites:

1. Operating System: Windows 10/11 (x64) with WSL 2 enabled, or native Ubuntu 22.04 LTS (jammy).
           Note: Ubuntu 22.04 is strictly required for binary compatibility with ROS 2 Humble.

2. Hardware: x64 CPU (16 GB+ RAM recommended; no discrete GPU required for local CPU physics).

____________________________________________________________________________

Replication Guide
1. Set Up Ubuntu 22.04 on WSL 2 (Windows Users)

   Open Windows PowerShell and install the specific Ubuntu 22.04 distribution

wsl --install -d Ubuntu-22.04

Launch Ubuntu 22.04, create your UNIX user credentials, and verify the OS codename returns jammy:

lsb_release -a

2. Install ROS 2 Humble

Inside the Ubuntu 22.04 terminal, add the ROS 2 GPG keys and repository, then install the desktop suite:

sudo apt update && sudo apt install curl software-properties-common -y

sudo add-apt-repository universe -y

sudo curl -sSL [https://raw.githubusercontent.com/ros/rosdistro/master/ros.key](https://raw.githubusercontent.com/ros/rosdistro/master/ros.key) -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] [http://packages.ros.org/ros2/ubuntu](http://packages.ros.org/ros2/ubuntu) $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update && sudo apt install ros-humble-desktop python3-colcon-common-extensions -y

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

source ~/.bashrc

3. Install MuJoCo, Robot Models & CPU ML Stack

Install Python dependencies, the MuJoCo physics engine, CPU-optimized PyTorch, and Stable Baselines3:

sudo apt install python3-pip git -y

pip3 install mujoco

pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

pip3 install "stable-baselines3[extra]"


Clone the DeepMind MuJoCo Menagerie repository to obtain the Unitree A1 physics model:

cd ~

git clone [https://github.com/google-deepmind/mujoco_menagerie.git](https://github.com/google-deepmind/mujoco_menagerie.git)


4. Clone and Build the ROS 2 Workspace

Clone this repository into your ROS 2 workspace src directory and build with colcon:


mkdir -p ~/ros2_ws/src

cd ~/ros2_ws/src

git clone git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git .

cd ~/ros2_ws

colcon build --symlink-install

echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

source ~/.bashrc


Usage
Run the Starter ROS 2 Controller Node

Launch the custom quadruped controller node to verify ROS 2 execution:

ros2 run quadruped_controller controller_node

Run the Standalone MuJoCo Quadruped Viewer

Verify physics rendering and WSLg graphical passthrough using the Unitree A1 model:

python3 ~/simulate_dog.py

(Tip: Double-click any body link in the viewer, then hold Ctrl + right-click and drag to apply external perturbation forces to the robot.)

----------------------------------------------------------------------------

Repository Structure

ros2_ws/
├── README.md
└── src/
    └── quadruped_controller/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        └── quadruped_controller/
            ├── __init__.py
            └── controller_node.py

----------------------------------------------------------------------------
Roadmap 

Configure WSL 2 (Ubuntu 22.04) and ROS 2 Humble environment

[1] Validate MuJoCo physics simulation with Unitree A1 URDF/MJCF model

[2] Initialize quadruped_controller ROS 2 Python package

[3] Bridge ROS 2 joint command/state topics with the MuJoCo simulation loop

[4] Implement a PD standing controller for static balance

[5] Train a PPO locomotion policy in MuJoCo and deploy via ROS 2 node





