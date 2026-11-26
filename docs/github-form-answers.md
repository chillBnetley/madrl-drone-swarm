# Subscription form — answers to paste

Repository to select: `madrl-drone-swarm` (it must be **public** and installed on the
GitHub app, otherwise it does not appear in the list).

## 1. Tell us about the project's reach and impact

> `madrl-drone-swarm` is the open-source release of my final-year engineering project
> (PFA) on multi-agent deep reinforcement learning for quadrotor swarms. It packages in
> one reproducible place three things that are usually scattered: (1) a Gazebo Sim +
> ROS 2 Humble X500 simulation setup including the integration fixes that are actually
> required to make it run at all, (2) classical baselines — a LiDAR-supervised square
> trajectory without any odometry source, and a PX4 SITL offboard state machine, and
> (3) a Gymnasium-compatible swarm environment with PPO and MAPPO training scripts that
> share the exact observation/action contract used by the ROS 2 nodes, so a policy
> trained offline can be evaluated in Gazebo unchanged.
>
> The gap it fills: today's open drone-RL code is either a purely kinematic toy
> environment with no robotics stack, or a full autopilot stack with no learning baseline
> and no documentation of the integration traps. In practice newcomers lose weeks on
> problems such as the missing `<robotNamespace>` on Gazebo motor plugins — which
> silently routes every motor command to the wrong model so the drone never moves — or
> the Humble-only requirement of declaring both `ros_topic_name` and `gz_topic_name` in
> `ros_gz_bridge`, or the fact that `ros-humble-ros-gz-sim` is not pulled in by
> `ros-humble-ros-gz`. All of these are documented and fixed in this repository, together
> with the architecture and experiment notes.
>
> Who it is for: students and research groups starting a swarm-RL project on ROS 2 +
> Gazebo + PX4, and anyone looking for a small, readable MAPPO implementation with a
> reproducible environment. Honest status: the project was just published, so traffic is
> modest — a handful of clones from my lab and classmates. I would rather report early,
> real numbers than inflated ones.

## 2. How will you use the subscription for your project?

> Daily engineering and research work on this repository: cross-checking ROS 2 Humble /
> Gazebo Harmonic / PX4 API changes, writing and reviewing the Python and C++ code (ROS 2
> nodes, Gazebo plugins, PPO/MAPPO training loop), and debugging build and runtime
> failures from simulator logs. Concretely, over the six months I need to finish the
> MAPPO training loop and its baseline comparison, keep the sim-to-ROS interface and the
> documentation accurate at every release, write the experiment notes, and prepare the
> material for my defence. The subscription replaces a manual copy/paste workflow with a
> proper code-review and documentation loop, and lets me keep the repository usable for
> the students who continue the work after me.

## 3. Other info

> The repository is MIT-licensed and will stay public. It accompanies my PFA report, which
> I can send if useful. Contact: momenhzez46@gmail.com. Note: the project is at an early
> stage (single maintainer, first public release), so evaluation should be based on the
> documented scope above rather than on usage metrics.
