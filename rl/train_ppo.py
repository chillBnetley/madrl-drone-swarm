#!/usr/bin/env python3
"""PPO baseline on the kinematic swarm environment (shared policy, single-agent view).

    python rl/train_ppo.py --steps 200000 --n-envs 4 --out runs/ppo

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
    parser.add_argument("--steps", type=int, default=200_000, help="total timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="parallel environments")
    parser.add_argument("--agents", type=int, default=3, help="agents per environment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/ppo", help="log / checkpoint directory")
    return parser.parse_args()


def make_env(seed: int, n_agents: int):
    from stable_baselines3.common.monitor import Monitor

    def _factory():
        env = DroneSwarmEnv(EnvConfig(n_agents=n_agents, seed=seed)).as_single_agent()
        return Monitor(env)

    return _factory


def main() -> int:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError:
        print(
            "stable-baselines3 is not installed.\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.out, exist_ok=True)
    env_fns = [make_env(args.seed + i, args.agents) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=args.out,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(os.path.join(args.out, "ppo_x500_swarm"))
    print(f"saved {os.path.join(args.out, 'ppo_x500_swarm.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
