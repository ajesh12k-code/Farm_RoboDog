#!/usr/bin/env bash
# Launch the Atharv Nandi ROS 2 stack from a source tree (default of docker/start.sh). The stack lives as long as
# the physics (sim_runner): closing the MuJoCo viewer or Ctrl+C ends everything. Every other node is supervised:
# a crash (non-zero exit) is logged and the node restarted, up to NANDI_NODE_RESTARTS (3) times; a node that ran
# for NANDI_NODE_STABLE_S (60 s) before crashing gets its restart budget back. A clean exit (status 0, e.g.
# closing the Tk console) is not restarted.
#
# Same node table, order and restart policy as launch/nandi.launch.py (atharv_nandi/launch_plan.py; checked by
# tests/test_package_layout.py). Each node runs as `python3 -m atharv_nandi.nodes <node> --ros-args --params-file
# $NANDI_PARAMS_FILE` (default config/nandi.yaml): declared ROS parameters, a set NANDI_* env var still wins.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_RESTARTS="${NANDI_NODE_RESTARTS:-3}"
STABLE_S="${NANDI_NODE_STABLE_S:-60}"
PARAMS="${NANDI_PARAMS_FILE:-$ROOT/config/nandi.yaml}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

trap 'kill 0' EXIT

log() { echo "[launch_nandi] $*" >&2; }

# run_node <name>: one node in the foreground.
run_node() { python3 -m atharv_nandi.nodes "$1" --ros-args --params-file "$PARAMS"; }

# supervise <name>: run the node in the background, restart on crash with a log line.
supervise() {
  local name="$1"; shift
  (
    set +e
    local restarts=0 rc start
    while true; do
      start=$SECONDS
      run_node "$name"
      rc=$?
      if [ "$rc" -eq 0 ]; then
        log "$name exited cleanly; not restarting"
        exit 0
      fi
      if [ $((SECONDS - start)) -ge "$STABLE_S" ]; then restarts=0; fi
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        log "$name exited with status $rc; giving up after $MAX_RESTARTS restarts"
        exit "$rc"
      fi
      restarts=$((restarts + 1))
      log "$name exited with status $rc; restart $restarts/$MAX_RESTARTS in 2 s"
      sleep 2
    done
  ) &
}

# A fresh stack starts un-latched; restarting only the safety_supervisor node keeps the e-stop latch.
# Same path the supervisor resolves: a set NANDI_ESTOP_STATE, else estop_state in $PARAMS, else the /tmp default.
python3 -m atharv_nandi.launch_plan clear-estop "$PARAMS"

# stage 0: services and pilot nodes
supervise work_envelope_service
supervise locomotion_service
supervise harvest_ik_service
supervise stability_monitor
supervise safety_supervisor
supervise mission_executor
supervise perception_node
supervise telemetry_bridge
supervise run_recorder
sleep 1
# stage 1: the physics (the stack's lifetime) and the browser console (Three.js + WebSocket) on
# :${NANDI_WEB_PORT:-8080}; no X display needed for the console.
LIBGL_ALWAYS_SOFTWARE=1 run_node sim_runner &
SIM_PID=$!
supervise web_bridge
sleep 1
# stage 2: the Tk console is optional: closing it must not take down the sim and the web UI.
supervise control_panel_gui
# The stack lives as long as the physics (closing the MuJoCo viewer or Ctrl+C ends everything).
wait "$SIM_PID"
