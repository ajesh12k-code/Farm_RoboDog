#!/usr/bin/env bash
# Build (if needed) and run the Atharv Nandi sim, then open
#   http://localhost:8080/                                     Three.js web console (fast)
#   http://localhost:6080/vnc.html?autoconnect=1&resize=scale  full desktop (MuJoCo viewer + Tk panel)
#
#   docker/run.sh                                  # full stack
#   docker/run.sh shell                            # virtual desktop + bash only
#   NANDI_CROPS=10 NANDI_TERRAIN=flat docker/run.sh   # world knobs are passed through when set
#   NANDI_WORLD=orchard_demo NANDI_NOVNC=0 docker/run.sh # orchard world, no noVNC desktop: person actors on
#   NANDI_OPERATOR_TOKEN=... NANDI_VIEWER_TOKEN=... docker/run.sh   # console auth (see README)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t atharv-nandi-sim -f "$ROOT/docker/Dockerfile" "$ROOT"
docker rm -f atharv-nandi >/dev/null 2>&1 || true
# Ports bind to localhost only: noVNC has no password and gives a desktop with the repo mounted rw.
# Args go to start.sh (e.g. "shell"); passed bare they would replace the image CMD.
exec docker run --rm -it --name atharv-nandi -p 127.0.0.1:6080:6080 -p 127.0.0.1:8080:8080 --shm-size=1g \
    -e NANDI_CROPS -e NANDI_TERRAIN -e NANDI_SEED -e NANDI_WORLD -e NANDI_ACTORS -e NANDI_NOVNC \
    -e NANDI_OPERATOR_TOKEN -e NANDI_VIEWER_TOKEN -e NANDI_FIELD_MODE -e NANDI_VNC_PASSWORD -e NANDI_WEB_ALLOWED_HOSTS -e NANDI_AUDIT_LOG -e NANDI_ESTOP_STATE -e NANDI_LAUNCH -e NANDI_PARAMS_FILE \
    -v "$ROOT:/ws" atharv-nandi-sim /usr/local/bin/start.sh "$@"
