#!/usr/bin/env bash
# Sanity check, then PPO, then MAPPO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python rl/check_env.py
python rl/train_ppo.py --steps 200000 --n-envs 4 --out runs/ppo
python rl/train_mappo.py --updates 200 --out runs/mappo
