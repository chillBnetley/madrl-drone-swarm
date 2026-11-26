#!/usr/bin/env python3
"""MAPPO (Multi-Agent PPO, shared parameters + centralised critic) skeleton.

This is the scaffolding for the multi-agent stage of the project: one shared
actor per agent and one critic that sees the concatenated swarm state.

    python rl/train_mappo.py --updates 200

Requires torch only (no other RL dependency).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from envs.drone_swarm_env import DroneSwarmEnv, EnvConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--rollout", type=int, default=128, help="steps per update")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="runs/mappo")
    return p.parse_args()


def build_models(obs_dim: int, act_dim: int, n_agents: int, lr: float):
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
            self.mu = nn.Linear(128, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        def forward(self, obs):
            hidden = self.body(obs)
            return torch.tanh(self.mu(hidden)), self.log_std.exp().expand_as(self.mu(hidden))

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim * n_agents, 256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(),
                nn.Linear(128, 1),
            )

        def forward(self, obs):
            return self.body(obs).squeeze(-1)

    actor = Actor()
    critic = Critic()
    return actor, critic, torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr
    )


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError:
        print("torch is not installed.\n  pip install -r requirements.txt", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    np.random.seed(0)

    env = DroneSwarmEnv(EnvConfig(n_agents=args.agents, seed=0))
    obs_dim, act_dim = env.obs_dim, env.act_dim
    actor, critic, optimiser = build_models(obs_dim, act_dim, args.agents, args.lr)
    os.makedirs(args.out, exist_ok=True)

    gamma, clip = 0.99, 0.2
    obs, _ = env.reset()
    for update in range(1, args.updates + 1):
        log_probs, rewards, values, masks = [], [], [], []
        for _ in range(args.rollout):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mu, std = actor(obs_t)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_probs.append(dist.log_prob(action).sum(-1))
            values.append(critic(obs_t.flatten().unsqueeze(0)).squeeze(0))

            obs, reward, terminated, truncated, info = env.step(action.numpy())
            rewards.append(torch.as_tensor(reward, dtype=torch.float32))
            masks.append(torch.as_tensor(~(terminated | truncated), dtype=torch.float32))
            if terminated.all() or truncated.all():
                obs, _ = env.reset()

        # Discounted return with the swarm mask
        returns, running = [], torch.zeros(args.agents)
        for step in reversed(range(args.rollout)):
            running = rewards[step] + gamma * running * masks[step]
            returns.insert(0, running.clone())
        returns_t = torch.stack(returns)

        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        advantage = returns_t - values_t
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(4):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mu, std = actor(obs_t)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(torch.as_tensor(obs_t, dtype=torch.float32)).sum(-1)
            ratio = torch.exp(new_log_probs - log_probs_t)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
            actor_loss = -torch.min(unclipped, clipped).mean()
            critic_loss = torch.nn.functional.mse_loss(values_t, returns_t)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            optimiser.step()
            break  # single inner epoch keeps the skeleton readable

        if update % 10 == 0:
            print(
                f"update {update:4d} | return {returns_t.sum().item():8.2f} "
                f"| loss {loss.item():7.3f}"
            )

    torch.save(
        {"actor": actor.state_dict(), "critic": critic.state_dict()},
        os.path.join(args.out, "mappo_x500_swarm.pt"),
    )
    print(f"saved {os.path.join(args.out, 'mappo_x500_swarm.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
