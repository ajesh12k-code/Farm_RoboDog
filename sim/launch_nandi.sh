#!/usr/bin/env bash
# Compatibility shim (WPR): images built before the ament_python restructure run /ws/sim/launch_nandi.sh from
# their baked-in start.sh. The launcher moved to launch/launch_nandi.sh (code now in atharv_nandi/).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/launch/launch_nandi.sh" "$@"
