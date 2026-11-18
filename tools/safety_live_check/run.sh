#!/usr/bin/env bash
# Headless full-stack live check inside a throwaway nandi-sim container. $1 = scenario (main|fence|fall)
source /opt/ros/humble/setup.bash
export NANDI_HEADLESS=1 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=$((RANDOM % 100 + 1))
cd /ws
export PYTHONPATH="/ws${PYTHONPATH:+:$PYTHONPATH}"
SC=${1:-main}
if [ "$SC" = fence ]; then
  printf 'geofence:\n  polygon: [[-0.5, -1.0], [0.35, -1.0], [0.35, 1.0], [-0.5, 1.0]]\n' > /tmp/fence.yaml
  export NANDI_SAFETY_CONFIG=/tmp/fence.yaml
fi
python3 -m atharv_nandi.nodes safety_supervisor > /tmp/sup.log 2>&1 &
if [ "$SC" != fall ]; then
  python3 -m atharv_nandi.nodes locomotion_service > /tmp/loco.log 2>&1 &
  [ "${HARVEST:-0}" = 1 ] && python3 -m atharv_nandi.nodes harvest_ik_service > /tmp/harv.log 2>&1 &
  python3 -m atharv_nandi.nodes sim_runner > /tmp/sim.log 2>&1 &
fi
sleep 4
python3 -u /ws/tools/safety_live_check/check.py "$SC"
rc=$?
echo "--- sup.log"; tail -n 8 /tmp/sup.log; echo "--- sim.log"; tail -n 6 /tmp/sim.log 2>/dev/null; echo "--- loco.log"; tail -n 6 /tmp/loco.log 2>/dev/null
kill 0 2>/dev/null
exit $rc
