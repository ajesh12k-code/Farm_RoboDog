#!/usr/bin/env bash
# Boot a virtual desktop, expose it on noVNC :6080, then launch the Atharv Nandi stack (web console on :8080).
set -e
source /opt/ros/humble/setup.bash
# Inside the container the bridge must listen on all interfaces; run.sh publishes it on 127.0.0.1 only.
export NANDI_WEB_HOST="${NANDI_WEB_HOST:-0.0.0.0}"

Xvfb :1 -screen 0 1600x900x24 -nolisten tcp &
sleep 1
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :1 -forever -shared -nopw -quiet -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
websockify --web /usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
echo "Web console:  http://localhost:8080/   (Three.js, low latency)"
echo "noVNC:        http://localhost:6080/vnc.html?autoconnect=1&resize=scale   (MuJoCo viewer + Tk panel)"

if [ "${1:-}" = "shell" ]; then exec bash; fi
exec bash /ws/sim/launch_nandi.sh
