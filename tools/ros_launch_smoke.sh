#!/usr/bin/env bash
# WPR acceptance smoke (issues #102-#104), inside the atharv-nandi-sim image (capped runner):
#   colcon build of the ament_python package -> source the install -> `ros2 launch atharv_nandi nandi.launch.py`
#   headless for SMOKE_S (20) s -> every node up, `ros2 param list` per node, /healthz, then an e-stop latch that
#   survives a safety_supervisor restart.
#   tools/ros_launch_smoke.sh [build|launch|all] [installed|source|bash]
# `build` and `launch` can run in separate capped runs: the install goes to /ws/.colcon (git-ignored).

MODE="${1:-all}"; SRC_MODE="${2:-installed}"
SMOKE_S="${SMOKE_S:-20}"
WS=/ws; OUT=$WS/.colcon
source /opt/ros/humble/setup.bash
fail() { echo "SMOKE FAIL: $*"; exit 1; }
if [ "$MODE" = build ] || [ "$MODE" = all ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"
  t0=$SECONDS
  (cd "$OUT" && colcon build --base-paths "$WS" --build-base "$OUT/build" --install-base "$OUT/install" \
      --event-handlers console_direct- >"$OUT/build.log" 2>&1) || { tail -40 "$OUT/build.log"; fail "colcon build"; }
  echo "colcon build ok in $((SECONDS - t0)) s"
  ls "$OUT/install/atharv_nandi/lib/atharv_nandi" | tr '\n' ' '; echo
  test -f "$OUT/install/atharv_nandi/share/atharv_nandi/config/nandi.yaml" || fail "config not installed"
  test -f "$OUT/install/atharv_nandi/share/atharv_nandi/launch/nandi.launch.py" || fail "launch not installed"
  find "$OUT/install" -path "*site-packages/atharv_nandi/sim/models/AtharvNandi/atharv_nandi.xml" | grep -q . \
    || fail "models not installed as package data"
  find "$OUT/install" -path "*site-packages/atharv_nandi/web/index.html" | grep -q . || fail "web not installed"
fi
[ "$MODE" = build ] && exit 0

export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=$((RANDOM % 100 + 1)) NANDI_HEADLESS=1 NANDI_WEB_PORT=8080
export NANDI_ESTOP_STATE=/tmp/nandi_estop_latch.json
LOG=/tmp/launch.log
if [ "$SRC_MODE" = installed ]; then
  source "$OUT/install/setup.bash"
  cd /tmp   # nothing from the source tree on sys.path: the installed copy and its share dir are used

  python3 -c "import atharv_nandi, atharv_nandi.paths as p; print('package', atharv_nandi.__file__); print('data root', p.REPO)"
  ros2 launch atharv_nandi nandi.launch.py gui:=false >"$LOG" 2>&1 &
elif [ "$SRC_MODE" = source ]; then
  export PYTHONPATH="$WS${PYTHONPATH:+:$PYTHONPATH}"
  cd "$WS"
  ros2 launch "$WS/launch/nandi.launch.py" gui:=false >"$LOG" 2>&1 &
else   # bash: the default launcher of docker/start.sh
  cd "$WS"
  bash "$WS/launch/launch_nandi.sh" >"$LOG" 2>&1 &
fi
LPID=$!
ros2 daemon start >/dev/null 2>&1   # discovery runs while the stack comes up
sleep "$SMOKE_S"
echo "--- nodes after ${SMOKE_S} s"
for _ in 1 2 3; do ros2 node list 2>/dev/null | sort -u > /tmp/nodes.txt; [ "$(grep -c . /tmp/nodes.txt)" -ge 11 ] && break; sleep 3; done
cat /tmp/nodes.txt
want="nandi_harvest_ik_service nandi_locomotion_service nandi_mission_executor nandi_perception nandi_run_recorder
nandi_safety_supervisor nandi_sim_runner nandi_stability_monitor nandi_telemetry_bridge nandi_web_bridge work_envelope_service"
missing=""
for n in $want; do grep -qx "/$n" /tmp/nodes.txt || missing="$missing $n"; done
echo "--- params (count per node; world / seed / one node-specific value)"
for n in $want; do
  c=$(timeout 10 ros2 param list "/$n" 2>/dev/null | grep -c .)
  echo "$n params=$c"
done
echo "web_bridge web_port=$(timeout 10 ros2 param get /nandi_web_bridge web_port 2>/dev/null)"
echo "sim_runner world_id=$(timeout 10 ros2 param get /nandi_sim_runner world_id 2>/dev/null)"
echo "--- healthz"
curl -fsS "http://127.0.0.1:8080/healthz" && echo || echo "healthz not ok"
echo "--- e-stop latch across a safety_supervisor restart"
timeout 10 ros2 service call /nandi/estop std_srvs/srv/Trigger >/dev/null 2>&1
sleep 1
[ -f "$NANDI_ESTOP_STATE" ] && echo "latch file written: $(cat "$NANDI_ESTOP_STATE")" || echo "no latch file"
pkill -f "safety_supervisor --ros-args" ; sleep 4
grep -c "safety_supervisor.*restart 1/3" "$LOG" | sed 's/^/supervisor restart log lines: /'
[ -f "$NANDI_ESTOP_STATE" ] && echo "latch file kept after restart" || echo "latch file LOST"
st=$(timeout 6 ros2 topic echo --once /nandi/safety_status 2>/dev/null | head -c 600)
echo "safety_status after restart: ${st:0:300}"
kill -INT "$LPID" 2>/dev/null; sleep 3; kill -9 "$LPID" 2>/dev/null
cp "$LOG" "$OUT/launch-$SRC_MODE.log" 2>/dev/null
echo "--- launch log (head/tail)"
head -5 "$LOG"; echo ...; grep -E "launch_nandi|ERROR|Traceback" "$LOG" | head -20
[ -z "$missing" ] || fail "nodes missing:$missing"
echo "SMOKE OK"
