# Atharv Nandi sim stack: ROS 2 Humble + MuJoCo + Tk. Browser UIs: Three.js web console on :8080
# (atharv_nandi/nodes/web_bridge.py, websockets 13) and the full X desktop via Xvfb -> x11vnc -> noVNC on :6080.
# WPR: the repo (mounted at /ws) is one ament_python package, atharv_nandi. PYTHONPATH=/ws makes it and tools/
# importable from the source tree; `colcon build` (python3-colcon-common-extensions, in ros-base) installs it.
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MUJOCO_GL=glfw \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-tk \
        ros-humble-geometry-msgs ros-humble-std-msgs ros-humble-std-srvs ros-humble-sensor-msgs \
        xvfb x11vnc fluxbox novnc websockify xterm curl \
        libgl1 libgl1-mesa-dri libglx-mesa0 libegl1 libglfw3 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# numpy<2 keeps ROS Humble's apt-built Python bindings happy. torch/SB3 omitted: the trained
# policy zip isn't in the repo, and locomotion_service falls back to the tripod gait without it.
RUN pip3 install --no-cache-dir "numpy>=1.24,<2" "mujoco>=3.3,<3.4" "websockets>=13.1,<14" "pytest>=7"
# WPCAP: OMPL 2.0.1 Python bindings (manylinux2014 aarch64/x86_64 wheels for cp310-cp313, no apt deps)
# for arm motion planning (atharv_nandi/core/planning/), and pytest-cov for the coverage gate (.coveragerc).
# ros2_control is not installed: the HAL in atharv_nandi/core/hal/ is a ros2_control-style Python layer
# (option B of docs/hal-sim-to-real.md) with the same interface names, so the C++ stack can come later.
RUN pip3 install --no-cache-dir "ompl==2.0.1" "pytest-cov>=5,<7"
# WP34 (OQ-33): py_trees 2.2.3 (BSD-3, pure Python, = ROS 2 Humble apt) is the workflow runtime (atharv_nandi/core/workflow/).
RUN pip3 install --no-cache-dir "py_trees==2.2.3"

COPY docker/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# WPR: `atharv_nandi` (the package) and `tools` import from the repo root mounted at /ws.
ENV PYTHONPATH=/ws
WORKDIR /ws
EXPOSE 6080 8080
# Healthy = web bridge up and sim_qpos fresh (< 2 s); see GET /healthz.
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${NANDI_WEB_PORT:-8080}/healthz" >/dev/null || exit 1
CMD ["/usr/local/bin/start.sh"]
