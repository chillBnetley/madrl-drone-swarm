"""Unit tests for the swarm environment contract (numpy only, no torch, no ROS).

    python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))

from envs.drone_swarm_env import DroneSwarmEnv, EnvConfig  # noqa: E402


def make_env(**kwargs) -> DroneSwarmEnv:
    config = EnvConfig(seed=0, **kwargs)
    return DroneSwarmEnv(config)


def test_observation_shape_and_finiteness():
    env = make_env()
    obs, info = env.reset(seed=1)

    assert obs.shape == (env.n, env.obs_dim)
    assert env.obs_dim == 6 + env.cfg.n_rays
    assert np.all(np.isfinite(obs))
    assert info["agent_mask"].shape == (env.n,)


def test_step_returns_gymnasium_five_tuple():
    env = make_env()
    env.reset(seed=2)
    actions = np.zeros((env.n, env.act_dim))

    obs, rewards, terminated, truncated, info = env.step(actions)

    assert obs.shape == (env.n, env.obs_dim)
    assert rewards.shape == terminated.shape == truncated.shape == (env.n,)
    assert terminated.dtype == bool and truncated.dtype == bool
    assert {"reached", "success", "step"} <= set(info)


def test_raycast_hits_known_obstacle():
    env = make_env(n_obstacles=0, obstacles=[(2.0, 5.0, 0.5)])
    env._build_obstacles()  # noqa: SLF001 - geometry check on purpose

    # From (0, 5) looking along +x, the obstacle surface is at x = 1.5.
    assert env._raycast(np.array([0.0, 5.0]), 0.0) == pytest.approx(1.5)  # noqa: SLF001
    # Looking the other way there is nothing within range.
    assert env._raycast(np.array([0.0, 5.0]), math.pi) == pytest.approx(env.cfg.ray_range)  # noqa: SLF001


def test_raycast_is_capped_by_range():
    env = make_env(n_obstacles=0, obstacles=[(5.0, 5.0, 0.5)])
    env._build_obstacles()  # noqa: SLF001

    # The surface is at 4.5 m, well past ray_range = 3.0 m.
    assert env._raycast(np.array([0.0, 5.0]), 0.0) == pytest.approx(env.cfg.ray_range)  # noqa: SLF001


def test_goal_inside_tolerance_terminates_with_bonus():
    env = make_env(n_obstacles=0, obstacles=[], goal_tolerance=0.35)
    env.reset(seed=3)

    env.pos = np.tile([5.0, 5.0], (env.n, 1))
    env.vel = np.zeros((env.n, 2))
    env.goals = np.tile([5.2, 5.0], (env.n, 1))

    _, rewards, terminated, _, info = env.step(np.zeros((env.n, env.act_dim)))

    assert terminated.all()
    assert info["reached"].all()
    assert info["success"] is True
    assert np.all(rewards >= 20.0)


def test_horizon_truncates_every_agent():
    env = make_env(horizon=5, n_obstacles=0, obstacles=[])
    env.reset(seed=4)
    env.pos = np.tile([1.0, 1.0], (env.n, 1))
    env.goals = np.tile([9.0, 9.0], (env.n, 1))

    truncated = np.zeros(env.n, dtype=bool)
    for _ in range(5):
        _, _, _, truncated, _ = env.step(np.ones((env.n, env.act_dim)))

    assert truncated.all()


def test_agent_stays_inside_the_area():
    env = make_env(n_obstacles=0, obstacles=[])
    env.reset(seed=5)
    env.pos = np.tile([9.9, 9.9], (env.n, 1))
    env.goals = np.tile([10.0, 10.0], (env.n, 1))

    for _ in range(20):
        env.step(np.ones((env.n, env.act_dim)))

    assert np.all(env.pos >= 0.0) and np.all(env.pos <= env.cfg.area)


def test_same_seed_gives_same_episode():
    first, _ = make_env().reset(seed=7)
    second, _ = make_env().reset(seed=7)

    np.testing.assert_allclose(first, second)


def test_single_agent_wrapper_contract():
    env = make_env()
    wrapped = env.as_single_agent()

    obs, _ = wrapped.reset(seed=8)
    assert obs.shape == (env.obs_dim,)

    obs, reward, done, info_truncated, _ = wrapped.step(np.zeros(env.act_dim))
    assert obs.shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info_truncated is False


def test_actions_are_clipped():
    env = make_env(n_obstacles=0, obstacles=[])
    env.reset(seed=9)
    env.pos = np.tile([5.0, 5.0], (env.n, 1))
    env.goals = np.tile([9.0, 5.0], (env.n, 1))

    for _ in range(5):
        env.step(np.full((env.n, env.act_dim), 50.0))

    assert np.all(np.abs(env.vel) <= env.cfg.max_speed + 1e-9)
