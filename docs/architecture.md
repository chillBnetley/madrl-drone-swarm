# Architecture

```
        +--------------------+        /x500/cmd_vel         +----------------------+
        |  ROS 2 node        | ---------------------------> | Gazebo Sim           |
        |  square_no_odom    |                              | X500 + LiDAR + IMU   |
        +--------------------+ <--------------------------- +----------------------+
                 ^   /x500/imu, /x500/scan                   (robotNamespace=x500)
                 |                                                |
                 |                    ros_gz_bridge               |
                 +------------------------------------------------+

  Learning side (no ROS dependency):

        +----------------------+      [v_cmd, w_cmd]     +--------------------+
        | DroneSwarmEnv        | ---------------------> | PPO / MAPPO        |
        | N agents, 2-D        | <--------------------- | shared policy      |
        +----------------------+   [x, y, vx, vy, dx, dy, rays]
```

Three layers, one contract:

| Layer | Entry point | Runs on |
|---|---|---|
| Simulation | `sim/worlds/x500_world.sdf` | Gazebo Sim (Harmonic) |
| Classical control | `square_no_odom.py`, `offboard_demo.py` | ROS 2 Humble / PX4 SITL |
| Learning | `rl/envs/drone_swarm_env.py` | any Python 3.10+ |

The observation vector is identical on both sides, which is what allows a policy
trained offline to be evaluated in Gazebo without touching the network:

```
obs = [x, y, vx, vy, dx_goal, dy_goal, ray_0 ... ray_(K-1)]
act = [v_cmd, w_cmd]
```

## Design choices

* **No odometry in the classical layer.** Pose comes from IMU dead reckoning, so the
  LiDAR is used only as a safety layer. This keeps the "no GPS, no odometry" constraint
  of the project.
* **Shared policy for MAPPO.** One actor for all agents plus one centralised critic on
  the concatenated swarm state: the parameter count stays O(1) in the number of drones.
* **Kinematic environment first.** The Gazebo loop runs at 1 kHz and is far too slow for
  millions of steps. The fast environment is the training ground, Gazebo is the
  validation ground.

## Message flow, step by step

1. `square_no_odom` publishes a `geometry_msgs/Twist` on `/x500/cmd_vel`.
2. `ros_gz_bridge` converts it to `gz.msgs.Twist` in the reverse direction (ROS -> GZ).
3. `MulticopterVelocityControl` turns the velocity into four motor speeds on
   `/x500/command/motor_speed`.
4. Gazebo integrates the dynamics and publishes IMU, LiDAR, NavSat and air pressure.
5. `ros_gz_bridge` converts them back to ROS 2 messages (GZ -> ROS).
6. The node integrates the IMU to estimate its position and closes its two PID loops.
