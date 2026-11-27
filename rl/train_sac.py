#!/usr/bin/env python3
"""SAC baseline on the kinematic swarm environment (single-agent view).

    python rl/train_sac.py --steps 300000 --out runs/sac

This is the script behind the 92 % success rate reported in the README (stage 1).
Requires stable-baselines3 (see requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from envs.drone_swarm_env import DroneSwarmEnv, EnvConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000, help="total timesteps")
    parser.add_argument("--agents", type=int, default=3, help="agents per environment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/sac", help="log / checkpoint directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print(
            "stable-baselines3 is not installed.\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.out, exist_ok=True)
    env = Monitor(DroneSwarmEnv(EnvConfig(n_agents=args.agents, seed=args.seed)).as_single_agent())

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=5_000,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=args.out,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(os.path.join(args.out, "sac_x500_swarm"))
    print(f"saved {os.path.join(args.out, 'sac_x500_swarm.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
