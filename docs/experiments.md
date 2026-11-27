# Experiments

Every number here comes from a run on the lab machine (Ubuntu 22.04, Python 3.10,
kinematic environment unless stated otherwise). Checkpoints are not committed: treat the
tables as something to reproduce, not as a benchmark.

## E1 — Square trajectory without odometry (classical layer)

Configuration: side 2.0 m, cruise speed 0.35 m/s, control loop 20 Hz, IMU dead reckoning,
yaw from the IMU quaternion.

Metric: position error at each corner, computed from the Gazebo ground truth
(`/x500/pose`), which is used **only for evaluation, never as feedback**.

Outcome: the four sides close within mission tolerance, and the error grows side after
side because of accumulated dead-reckoning drift. That drift is the motivation for the
learning layer. Raw logs are produced by the node itself:

```bash
ros2 run x500_navigation square_no_odom --ros-args -p csv_path:=/tmp/square_log.csv
```

## E2 — PPO on the kinematic swarm environment

| Item | Value |
|---|---|
| Environment | `DroneSwarmEnv`, 3 agents, 8 raycasts, 10 m x 10 m |
| Algorithm | PPO, MLP 128x128, lr 3e-4, gamma 0.99 |
| Budget | ~1.85 M steps |

Outcome: the policy converges — agents reach their goals and the average return
stabilises. Two failure modes were observed:

1. **Spinning reward hack.** Without an angular-rate penalty the agent turns in place to
   farm the progress shaping term. Fixed with `-1.5 * |w| * dt`.
2. **Corner trapping.** With four obstacles and no raycast the agent gets stuck in
   concave corners. The eight raycasts in the observation fix it.

PPO is the baseline, not the best result: `--steps 200000` is far below the budget the
good runs used, so a fresh clone will look worse than the table above.

## E3 — SAC, single-agent view

| Item | Value |
|---|---|
| Algorithm | SAC, MLP 256x256, lr 3e-4, buffer 200 k, learned after 5 k random steps |
| Script | `python rl/train_sac.py --steps 300000 --out runs/sac` |
| Success rate | **92 %** (stage 1) |

Stage 1 is the flat 10 m x 10 m world with four obstacles and no curriculum: the same
task the other runs use, evaluated with a greedy policy.

## E4 — MAPPO, and the run that broke

Shared actor, centralised critic, 3 agents, 128-step rollouts (`rl/configs/mappo.yaml`).

| Run | Success rate | What it was |
|---|---|---|
| v1 | **42 %** | first MAPPO configuration that produced usable behaviour |
| v2 | **8 %** | same environment, changed training configuration — success collapsed |
| v3 | *pending* | re-run of the v1 configuration; the number is being re-measured |

What the three runs say, and why v2 stays in this file:

* **The metric is a cliff.** Nothing in `rl/envs/` changed between v1 and v2; the policy
  went from reaching the goal in 2 of 5 episodes to roughly 1 in 12. That gap is not
  "slightly worse tuning", it is a different regime — and it means a single success
  percentage hides how close the policy is to falling off.
* **Sample budget favours the single-agent baseline.** SAC owns its whole buffer and
  learns one policy for one agent view. MAPPO shares one actor across three agents with a
  centralised critic that has to model all of them; at a few hundred thousand steps it
  has seen far less per-agent data. The honest reading is "MAPPO needs a bigger budget
  here", not "MAPPO is worse".
* **Keeping failures is the point.** Anyone reproducing this will hit a run that drops;
  knowing that v2 dropped without an environment change tells them to look at the
  training configuration and the seeds first.
* **No number is claimed for v3 yet.** It is re-run before being published, because a
  single seed on this task is not a result.

## Reproduce

```bash
python rl/check_env.py
python -m pytest tests -q
python rl/train_ppo.py --steps 200000 --seed 0 --out runs/ppo
python rl/train_sac.py --steps 300000 --seed 0 --out runs/sac
tensorboard --logdir runs
```
