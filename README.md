# MADRL Drone Swarm — Autonomous Navigation for an X500 Quadrotor

Reference implementation and research toolkit for my final-year project (PFA):
**Multi-Agent Deep Reinforcement Learning for autonomous navigation of a drone swarm**,
developed on the **X500** quadrotor model with **ROS 2 Humble**, **Gazebo Sim (Harmonic)**
and **PX4 SITL**.

The repository has three layers:

| Layer | Folder | What it does |
|---|---|---|
| Simulation | `sim/` | X500 SDF model, Gazebo world, `ros_gz_bridge` configuration |
| Classical control | `ros2_ws/` | ROS 2 nodes: square trajectory without odometry, PX4 offboard demo |
| Learning | `rl/` | Gymnasium environment + PPO / MAPPO training scripts |

## Why this project

Most open-source drone-learning code is either a pure Python toy environment with no
robotics stack, or a full autopilot stack with no learning baseline. This project keeps
the two sides separable: the same observation/action contract (`rl/envs/drone_swarm_env.py`)
is used by the fast kinematic environment during training and by the ROS 2 / Gazebo
pipeline during evaluation.

## Quick start (no ROS needed)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python rl/train_ppo.py --steps 200000 --n-envs 4
```

The training script writes TensorBoard logs and checkpoints to `runs/`.

## Full stack (ROS 2 Humble + Gazebo + PX4)

See [`docs/installation.md`](docs/installation.md) for the complete setup, the known
pitfalls (SDF `robotNamespace`, Humble bridge syntax, `ros_gz_sim` missing package) and
the exact commands.

```bash
# 1. Gazebo world with the X500
gz sim -r -v 4 sim/worlds/x500_world.sdf

# 2. Bridge Gazebo <-> ROS 2
ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:=$PWD/sim/bridge/x500_bridge.yaml

# 3. Square trajectory without odometry
ros2 run x500_navigation square_no_odom
```

PX4 SITL path (offboard control):

```bash
MicroXRCEAgent udp4 -p 8888      # terminal 1
ros2 run x500_navigation offboard_demo   # terminal 2
```

## Repository layout

```
sim/models/x500/model.sdf        X500 model with per-plugin <robotNamespace>
sim/worlds/x500_world.sdf        Empty world + model include + pose publisher
sim/bridge/x500_bridge.yaml      IMU, NavSat, 2D LiDAR, air pressure, pose -> ROS 2
ros2_ws/src/x500_navigation/     square_no_odom.py, offboard_demo.py
rl/envs/drone_swarm_env.py       Multi-agent kinematic environment (Gymnasium API)
rl/train_ppo.py                  Single-agent PPO baseline (stable-baselines3)
rl/train_mappo.py                Shared-parameter MAPPO skeleton (torch)
docs/                            Architecture, installation, experiments
```

## Results so far

* Classical layer: square trajectory closed without odometry, position error kept below
  the tolerance of the LiDAR-based correction loop (see `docs/experiments.md`).
* Learning layer: PPO converges on the 2-D kinematic environment (~1.85 M steps,
  see the learning curves in `docs/experiments.md`).

Multi-agent MAPPO on the full ROS 2 / Gazebo stack is the ongoing work.

## License

MIT — see [`LICENSE`](LICENSE).
