#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trap 'kill 0' EXIT

python3 "$DIR/work_envelope_service.py" &
python3 "$DIR/locomotion_service.py" &
python3 "$DIR/harvest_ik_service.py" &
sleep 1
LIBGL_ALWAYS_SOFTWARE=1 python3 "$DIR/sim_runner.py" &
sleep 1
python3 "$DIR/control_panel_gui.py"
