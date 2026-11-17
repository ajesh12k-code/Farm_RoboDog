#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trap 'kill 0' EXIT

python3 "$DIR/work_envelope_service.py" &
python3 "$DIR/locomotion_service.py" &
python3 "$DIR/harvest_ik_service.py" &
sleep 1
LIBGL_ALWAYS_SOFTWARE=1 python3 "$DIR/sim_runner.py" &
SIM_PID=$!
# Browser console (Three.js + WebSocket) on :${NANDI_WEB_PORT:-8080}; no X display needed.
python3 "$DIR/web_bridge.py" &
sleep 1
# The Tk console is optional: closing it must not take down the sim and the web UI.
python3 "$DIR/control_panel_gui.py" &
# The stack lives as long as the physics (closing the MuJoCo viewer or Ctrl+C ends everything).
wait "$SIM_PID"
