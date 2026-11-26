# Experiments

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

## E3 — MAPPO (ongoing)

Shared actor plus centralised critic, 3 agents, 128-step rollouts. The skeleton in
`rl/train_mappo.py` trains end to end; the hyperparameter search and the comparison with
the independent-PPO baseline are in progress.

## Reproduce

```bash
python rl/check_env.py
python rl/train_ppo.py --steps 200000 --seed 0 --out runs/ppo
tensorboard --logdir runs
```
