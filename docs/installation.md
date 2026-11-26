# Installation

Validated on **Ubuntu 22.04.5 (jammy)**, **ROS 2 Humble**, **Gazebo Sim 8.10.0 (Harmonic)**
and **PX4 SITL**.

## 1. System packages

```bash
sudo apt update
sudo apt install ros-humble-ros-gz \
                 ros-humble-ros-gz-bridge \
                 ros-humble-ros-gz-interfaces \
                 ros-humble-ros-gz-sim
```

`ros_gz_sim` is **not** pulled in by `ros-humble-ros-gz` alone; its absence is the first
error most people hit (`gz sim` integration missing).

## 2. Build the workspace

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 3. Run the simulation

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PWD/sim/plugins/build
gz sim -r -v 4 sim/worlds/x500_world.sdf
```

Second terminal:

```bash
ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:=$PWD/sim/bridge/x500_bridge.yaml
```

Checks:

```bash
ros2 topic hz /x500/imu
ros2 topic echo /x500/scan --once
```

## Known pitfalls (all fixed in this repository)

1. **Motor commands land on the wrong model.** With several models loaded, a motor plugin
   without `<robotNamespace>` inherits the namespace Gazebo registered last, e.g.
   `/x500_lidar_2d/command/motor_speed`. The drone then never reacts to `/x500/cmd_vel`.
   Fix: an explicit `<robotNamespace>x500</robotNamespace>` on each of the four motor
   plugins.
2. **The bridge silently ignores topics.** On Humble, `topic_name:` alone does nothing —
   both `ros_topic_name:` and `gz_topic_name:` are required. The short form is Jazzy only.
3. **`MotorFailurePlugin` does not exist** in a standalone Gazebo Harmonic installation;
   remove it from the model.
4. **Version mismatch between `ros_gz_interfaces` and Gazebo** produces
   `Failed to load system plugin`.

## 4. PX4 SITL path

```bash
MicroXRCEAgent udp4 -p 8888                 # terminal 1
ros2 run x500_navigation offboard_demo      # terminal 2
```

Port 8888 is the PX4 default (`uxrce_dds_client start -t udp -h 127.0.0.1 -p 8888`).
PX4 works in NED: climbing means a **negative** z, hence the hover altitude `-2.0` m.

## 5. Learning side only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python rl/check_env.py
python rl/train_ppo.py --steps 200000 --n-envs 4
```
