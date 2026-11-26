"""Multi-agent kinematic environment for the X500 swarm (Gymnasium-style API).

The environment deliberately has no ROS dependency: it runs anywhere (also on
Windows) so the learning pipeline can be developed and profiled fast. The
observation/action contract is the one implemented on the ROS 2 side:

    observation = [x, y, vx, vy, dx_goal, dy_goal, ray_0..ray_(K-1)]
    action      = [v_cmd, w_cmd]   with v_cmd in [-1, 1] m/s and w_cmd in [-1, 1] rad/s

`as_single_agent()` flattens the swarm to a single-agent view so a standard
PPO implementation can be trained on it (shared-policy baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # gymnasium is optional: the env also works with plain numpy loops
    import gymnasium as gym
    from gymnasium import spaces

    _GYM = True
except Exception:  # pragma: no cover
    gym = object  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _GYM = False


@dataclass
class EnvConfig:
    n_agents: int = 3
    area: float = 10.0           # square side, metres
    max_speed: float = 1.0       # m/s
    max_yaw_rate: float = 1.5    # rad/s
    dt: float = 0.1
    horizon: int = 600           # steps before truncation
    n_rays: int = 8
    ray_range: float = 3.0
    goal_tolerance: float = 0.35
    collision_radius: float = 0.45
    n_obstacles: int = 4
    seed: int = 0
    obstacles: list[tuple[float, float, float]] = field(default_factory=list)


class DroneSwarmEnv:
    """Decentralised swarm task: every agent reaches its own goal, avoiding the
    static obstacles and the other agents."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.n = self.cfg.n_agents
        self.obs_dim = 6 + self.cfg.n_rays
        self.act_dim = 2

        self.pos = np.zeros((self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.yaw = np.zeros(self.n)
        self.goals = np.zeros((self.n, 2))
        self.obstacles = np.zeros((0, 3))
        self.steps = 0
        self._reached = np.zeros(self.n, dtype=bool)

        if _GYM:
            self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), dtype=np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (self.obs_dim,), dtype=np.float32
            )

    # ------------------------------------------------------------------ helpers
    def _sample_free_point(self) -> np.ndarray:
        for _ in range(100):
            p = self.rng.uniform(0.5, self.cfg.area - 0.5, size=2)
            if self.obstacles.size == 0:
                return p
            distances = np.linalg.norm(self.obstacles[:, :2] - p, axis=1)
            if np.all(distances > self.obstacles[:, 2] + 0.8):
                return p
        return p

    def _build_obstacles(self) -> None:
        if self.cfg.obstacles:
            self.obstacles = np.asarray(self.cfg.obstacles, dtype=float)
            return
        centres = self.rng.uniform(2.0, self.cfg.area - 2.0, size=(self.cfg.n_obstacles, 2))
        radii = self.rng.uniform(0.3, 0.7, size=self.cfg.n_obstacles)
        self.obstacles = np.column_stack([centres, radii])

    def _raycast(self, origin: np.ndarray, angle: float) -> float:
        """Distance to the first obstacle along a ray (capped at ray_range)."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        best = self.cfg.ray_range
        for cx, cy, radius in self.obstacles:
            oc = origin - np.array([cx, cy])
            b = 2.0 * float(oc @ direction)
            c = float(oc @ oc) - radius * radius
            disc = b * b - 4.0 * c
            if disc < 0.0:
                continue
            sqrt_disc = np.sqrt(disc)
            for t in ((-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0):
                if t > 0.0:
                    best = min(best, t)
                    break
        return float(best)

    def _observation(self, i: int) -> np.ndarray:
        goal_delta = self.goals[i] - self.pos[i]
        rays = [
            self._raycast(self.pos[i], self.yaw[i] + 2.0 * np.pi * k / self.cfg.n_rays)
            / self.cfg.ray_range
            for k in range(self.cfg.n_rays)
        ]
        return np.concatenate([
            self.pos[i] / self.cfg.area,
            self.vel[i] / self.cfg.max_speed,
            goal_delta / self.cfg.area,
            np.asarray(rays, dtype=float),
        ]).astype(np.float32)

    def _inside_obstacle(self, point: np.ndarray, margin: float = 0.0) -> bool:
        if self.obstacles.size == 0:
            return False
        distances = np.linalg.norm(self.obstacles[:, :2] - point, axis=1)
        return bool(np.any(distances < self.obstacles[:, 2] + margin))

    # ------------------------------------------------------------------ Gym API
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._build_obstacles()

        self.pos = np.array([self._sample_free_point() for _ in range(self.n)])
        self.yaw = self.rng.uniform(-np.pi, np.pi, size=self.n)
        self.vel = np.zeros((self.n, 2))

        self.goals = np.zeros((self.n, 2))
        for i in range(self.n):
            for _ in range(100):
                candidate = self._sample_free_point()
                if np.linalg.norm(candidate - self.pos[i]) > 2.0:
                    self.goals[i] = candidate
                    break

        self.steps = 0
        self._reached = np.zeros(self.n, dtype=bool)
        obs = np.stack([self._observation(i) for i in range(self.n)])
        info = {"agent_mask": np.ones(self.n, dtype=bool)}
        return obs, info

    def step(self, actions):
        actions = np.asarray(actions, dtype=float).reshape(self.n, self.act_dim)
        actions = np.clip(actions, -1.0, 1.0)

        v_cmd = actions[:, 0] * self.cfg.max_speed
        w_cmd = actions[:, 1] * self.cfg.max_yaw_rate

        self.yaw = np.arctan2(
            np.sin(self.yaw + w_cmd * self.cfg.dt),
            np.cos(self.yaw + w_cmd * self.cfg.dt),
        )
        self.vel = np.column_stack([np.cos(self.yaw), np.sin(self.yaw)]) * v_cmd[:, None]

        new_pos = self.pos + self.vel * self.cfg.dt
        new_pos = np.clip(new_pos, 0.0, self.cfg.area)

        rewards = np.zeros(self.n)
        terminated = np.ones(self.n, dtype=bool)

        for i in range(self.n):
            reward = 0.0
            previous_distance = float(np.linalg.norm(self.goals[i] - self.pos[i]))
            blocked = self._inside_obstacle(new_pos[i], self.cfg.collision_radius * 0.3)
            if blocked:
                reward -= 5.0
                new_pos[i] = self.pos[i]
            else:
                distance = float(np.linalg.norm(self.goals[i] - new_pos[i]))
                reward += 1.5 * (previous_distance - distance)   # progress shaping
                reward -= 1.5 * abs(w_cmd[i]) * self.cfg.dt       # penalise spinning
                if distance < self.cfg.goal_tolerance:
                    reward += 25.0
                    self._reached[i] = True
                else:
                    terminated[i] = False

            # Agent-agent separation
            for j in range(self.n):
                if i == j:
                    continue
                if np.linalg.norm(new_pos[i] - new_pos[j]) < self.cfg.collision_radius:
                    reward -= 2.0

            rewards[i] = reward

        self.pos = new_pos
        self.steps += 1

        obs = np.stack([self._observation(i) for i in range(self.n)])
        truncated = np.full(self.n, self.steps >= self.cfg.horizon, dtype=bool)
        info = {
            "reached": self._reached.copy(),
            "success": bool(self._reached.all()),
            "step": self.steps,
        }
        return obs, rewards, terminated, truncated, info

    def as_single_agent(self):
        return SingleAgentWrapper(self)


class SingleAgentWrapper:
    """Flatten the swarm so a standard single-agent PPO can learn a shared policy
    over the agents: one agent is controlled at a time, cycling through the swarm."""

    def __init__(self, env: DroneSwarmEnv):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._index = 0
        self._last_obs = None
        self._last_reward = 0.0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._index = 0
        return obs[0], info

    def step(self, action):
        n = self.env.n
        actions = np.zeros((n, self.env.act_dim))
        actions[self._index] = action
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        reward = float(rewards[self._index])
        self._index = (self._index + 1) % n
        done = bool(terminated.all() or truncated.all())
        return obs[self._index], reward, done, False, info
