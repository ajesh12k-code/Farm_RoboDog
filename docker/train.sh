#!/usr/bin/env bash
# Headless PPO training of the 23-DOF arm-assisted balance policy inside Docker.
#
#   docker/train.sh                 # run rl_training/train_nandi_balance.py (auto-resumes)
#   docker/train.sh python3 X.py    # run any other command in rl_training/ with the training image
#
# Checkpoints land in rl_training/models/ on the host (repo is bind-mounted). Training is CPU-heavy
# (4 MuJoCo worker processes, 1.2 M timesteps): run it on a Linux workstation or cloud VM, not a laptop.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build -t atharv-nandi-sim -f "$ROOT/docker/Dockerfile" "$ROOT"
docker build -t atharv-nandi-train -f "$ROOT/docker/Dockerfile.train" "$ROOT/docker"
mkdir -p "$ROOT/rl_training/models"

if [ "$#" -eq 0 ]; then
    set -- python3 train_nandi_balance.py
fi

# Only ask for a TTY when there is one, so the script also works under nohup / tmux / CI.
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS=(-it); fi

exec docker run --rm "${TTY_FLAGS[@]}" --name atharv-nandi-train \
    --shm-size=2g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -v "$ROOT:/ws" -w /ws/rl_training \
    atharv-nandi-train "$@"
