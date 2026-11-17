# Atharv Nandi sim stack: ROS 2 Humble + MuJoCo + Tk. Browser UIs: Three.js web console on :8080
# (sim/web_bridge.py, websockets 13) and the full X desktop via Xvfb -> x11vnc -> noVNC on :6080.
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
        xvfb x11vnc fluxbox novnc websockify xterm \
        libgl1 libgl1-mesa-dri libglx-mesa0 libegl1 libglfw3 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# numpy<2 keeps ROS Humble's apt-built Python bindings happy. torch/SB3 omitted: the trained
# policy zip isn't in the repo, and locomotion_service falls back to the tripod gait without it.
RUN pip3 install --no-cache-dir "numpy>=1.24,<2" "mujoco>=3.3,<3.4" "websockets>=13.1,<14" "pytest>=7"

COPY docker/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /ws
EXPOSE 6080 8080
CMD ["/usr/local/bin/start.sh"]
