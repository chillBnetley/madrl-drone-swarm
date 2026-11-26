#!/usr/bin/env bash
# Start Gazebo and the ros_gz bridge in one terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-$ROOT/sim/plugins/build}"

gz sim -r -v 4 "$ROOT/sim/worlds/x500_world.sdf" &
GZ_PID=$!

sleep 3

ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:="$ROOT/sim/bridge/x500_bridge.yaml" &
BRIDGE_PID=$!

trap 'kill "$GZ_PID" "$BRIDGE_PID" 2>/dev/null || true' EXIT
wait
