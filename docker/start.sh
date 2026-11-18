#!/usr/bin/env bash
# Boot a virtual desktop, expose it on noVNC :6080, then launch the Atharv Nandi stack (web console on :8080).
#   NANDI_VNC_PASSWORD=...  x11vnc asks for this password (default: no password, ports bound to 127.0.0.1)
#   NANDI_FIELD_MODE=1      no VNC/noVNC desktop at all; the web console then requires tokens
#   NANDI_NOVNC=0           no VNC/noVNC desktop (console auth unchanged); lets orchard worlds compose person actors
#   NANDI_LAUNCH=ros2       start the stack with `ros2 launch` (launch/nandi.launch.py) instead of launch/launch_nandi.sh
# Exports NANDI_NOVNC_UP=0|1 (the decision below) for the stack: atharv_nandi/sim/world/actors.py reads it for the privacy
# lock-out (MVP-SPEC 5.4). Keep the patterns in step with actors.FIELD_MODE_ON / NOVNC_OFF.
set -e
source /opt/ros/humble/setup.bash
# Inside the container the bridge must listen on all interfaces; run.sh publishes it on 127.0.0.1 only.
export NANDI_WEB_HOST="${NANDI_WEB_HOST:-0.0.0.0}"

# docker restart keeps /tmp, so a stale X lock from the previous boot would stop Xvfb (and then the MuJoCo viewer).
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
Xvfb :1 -screen 0 1600x900x24 -nolisten tcp &
sleep 1
fluxbox >/tmp/fluxbox.log 2>&1 &
case "${NANDI_FIELD_MODE:-0}" in 1|true|yes|on) FIELD=1 ;; *) FIELD=0 ;; esac
case "${NANDI_NOVNC:-1}" in 0|false|no|off) NOVNC=0 ;; *) NOVNC=1 ;; esac
if [ "$FIELD" = 1 ]; then NOVNC=0; fi
export NANDI_NOVNC_UP="$NOVNC"
if [ "$FIELD" = 1 ]; then
  echo "Field mode:   noVNC desktop disabled; web console requires NANDI_OPERATOR_TOKEN"
elif [ "$NOVNC" = 0 ]; then
  echo "noVNC:        disabled (NANDI_NOVNC=$NANDI_NOVNC); person actors may be composed"
else
  if [ -n "${NANDI_VNC_PASSWORD:-}" ]; then
    x11vnc -storepasswd "$NANDI_VNC_PASSWORD" /tmp/x11vnc.pass >/dev/null 2>&1
    VNC_AUTH=(-rfbauth /tmp/x11vnc.pass)
  else
    VNC_AUTH=(-nopw)
  fi
  x11vnc -display :1 -forever -shared "${VNC_AUTH[@]}" -quiet -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
  websockify --web /usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
  echo "noVNC:        http://localhost:6080/vnc.html?autoconnect=1&resize=scale   (MuJoCo viewer + Tk panel)"
fi
echo "Web console:  http://localhost:8080/   (Three.js, low latency; health: /healthz)"

# The package (atharv_nandi/) and the dev tools (tools/) import from the repo root (also ENV in docker/Dockerfile).
case ":${PYTHONPATH:-}:" in *:/ws:*) ;; *) export PYTHONPATH="/ws${PYTHONPATH:+:$PYTHONPATH}" ;; esac
if [ "${1:-}" = "shell" ]; then exec bash; fi
# NANDI_LAUNCH=ros2: the same bringup through `ros2 launch` (launch/nandi.launch.py, nodes from the source tree);
# default: the bash supervisor launch/launch_nandi.sh. Both load config/nandi.yaml (NANDI_PARAMS_FILE overrides).
if [ "${NANDI_LAUNCH:-bash}" = "ros2" ]; then
  exec ros2 launch /ws/launch/nandi.launch.py params_file:="${NANDI_PARAMS_FILE:-/ws/config/nandi.yaml}"
fi
exec bash /ws/launch/launch_nandi.sh
