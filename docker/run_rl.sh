#!/usr/bin/env bash
# Run the full Atharv Nandi sim with torch + Stable-Baselines3 available, so the locomotion node can load
# a trained policy from rl_training/models/ and the RL Balance mode can be toggled on.
# Same UI as docker/run.sh: browser UI http://localhost:8080, noVNC http://localhost:6080.
#
#   docker/run_rl.sh          # full stack
#   docker/run_rl.sh shell    # virtual desktop + bash only (e.g. to run rl_training/evaluate_*.py)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build -t atharv-nandi-sim -f "$ROOT/docker/Dockerfile" "$ROOT"
docker build -t atharv-nandi-train -f "$ROOT/docker/Dockerfile.train" "$ROOT/docker"

MODELS="$ROOT/rl_training/models"
# nandi_balance_* = current names; farmdog6_balance_* = pre-rename archives, which still load.
if ! compgen -G "$MODELS/nandi_balance_*.zip" >/dev/null && ! compgen -G "$MODELS/farmdog6_balance_*.zip" >/dev/null; then
    echo "NOTE: no policy in rl_training/models/ (nandi_balance_final.zip or nandi_balance_*_steps.zip)."
    echo "      The sim will run on the Tripod gait only. Train one with docker/train.sh."
fi

docker rm -f atharv-nandi >/dev/null 2>&1 || true
exec docker run --rm -it --name atharv-nandi \
    -p 127.0.0.1:6080:6080 -p 127.0.0.1:8080:8080 --shm-size=1g \
    -e NANDI_CROPS -e NANDI_TERRAIN -e NANDI_SEED \
    -v "$ROOT:/ws" atharv-nandi-train /usr/local/bin/start.sh "$@"
