#!/usr/bin/env python3
"""PX4 SITL offboard demo for the X500.

State machine
    WAIT_FOR_TELEMETRY -> WARMUP (2 s) -> REQUEST_OFFBOARD -> WAIT_OFFBOARD
    -> REQUEST_ARM -> WAIT_ARM -> HOVER (5 s at z = -2.0 m) -> LAND -> COMPLETE

Prerequisites
    MicroXRCEAgent udp4 -p 8888        # uXRCE-DDS bridge, default PX4 port
    px4_msgs built in the workspace     # colcon build --packages-select px4_msgs

Run
    ros2 run x500_navigation offboard_demo

Notes
    * PX4 uses NED: a climb is a NEGATIVE z.
    * Offboard mode requires a setpoint stream of at least 2 Hz before the request;
      that is what the WARMUP phase is for.
"""

from __future__ import annotations

import enum

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

HOVER_Z = -2.0           # NED, so 2 m above the ground
WARMUP_SECONDS = 2.0
HOVER_SECONDS = 5.0
SETPOINT_RATE = 20.0     # Hz


class State(enum.Enum):
    WAIT_FOR_TELEMETRY = enum.auto()
    WARMUP = enum.auto()
    REQUEST_OFFBOARD = enum.auto()
    WAIT_OFFBOARD = enum.auto()
    REQUEST_ARM = enum.auto()
    WAIT_ARM = enum.auto()
    HOVER = enum.auto()
    LAND = enum.auto()
    COMPLETE = enum.auto()


def px4_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class OffboardDemo(Node):
    def __init__(self) -> None:
        super().__init__("offboard_demo")
        qos = px4_qos()

        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos
        )

        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.on_local_pos, qos
        )
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.on_att, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v2", self.on_status, qos)

        self.local_pos: VehicleLocalPosition | None = None
        self.attitude: VehicleAttitude | None = None
        self.status: VehicleStatus | None = None

        self.state = State.WAIT_FOR_TELEMETRY
        self.state_since = self.get_clock().now()
        self.setpoint = [0.0, 0.0, HOVER_Z]

        self.create_timer(1.0 / SETPOINT_RATE, self.step)
        self.get_logger().info("offboard_demo started, waiting for PX4 telemetry…")

    # ------------------------------------------------------------------ callbacks
    def on_local_pos(self, msg: VehicleLocalPosition) -> None:
        self.local_pos = msg

    def on_att(self, msg: VehicleAttitude) -> None:
        self.attitude = msg

    def on_status(self, msg: VehicleStatus) -> None:
        self.status = msg

    # ------------------------------------------------------------------ helpers
    def elapsed(self) -> float:
        return (self.get_clock().now() - self.state_since).nanoseconds * 1e-9

    def goto(self, state: State) -> None:
        self.get_logger().info(f"{self.state.name} -> {state.name}")
        self.state = state
        self.state_since = self.get_clock().now()

    def publish_command(self, command: int, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        for index, value in params.items():
            setattr(msg, index, float(value))
        self.cmd_pub.publish(msg)

    def publish_setpoint_stream(self) -> None:
        mode = OffboardControlMode()
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        mode.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.mode_pub.publish(mode)

        sp = TrajectorySetpoint()
        sp.position = [float(v) for v in self.setpoint]
        sp.yaw = 0.0
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(sp)

    # ------------------------------------------------------------------ main loop
    def step(self) -> None:
        # A setpoint stream is mandatory before and during offboard
        self.publish_setpoint_stream()

        if self.state is State.WAIT_FOR_TELEMETRY:
            if self.local_pos is not None and self.status is not None:
                self.setpoint[2] = float(self.local_pos.z)
                self.goto(State.WARMUP)

        elif self.state is State.WARMUP:
            if self.elapsed() > WARMUP_SECONDS:
                self.goto(State.REQUEST_OFFBOARD)

        elif self.state is State.REQUEST_OFFBOARD:
            # MAV_CMD_DO_SET_MODE, custom_mode 6 = PX4_CUSTOM_MAIN_MODE_OFFBOARD
            self.publish_command(
                176, param1=1.0, param2=6.0, param3=0.0, param4=0.0,
                param5=0.0, param6=0.0, param7=0.0,
            )
            self.goto(State.WAIT_OFFBOARD)

        elif self.state is State.WAIT_OFFBOARD:
            if self.status and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.goto(State.REQUEST_ARM)
            elif self.elapsed() > 5.0:
                self.get_logger().warn("offboard not confirmed, retrying")
                self.goto(State.REQUEST_OFFBOARD)

        elif self.state is State.REQUEST_ARM:
            self.publish_command(400)  # MAV_CMD_COMPONENT_ARM_DISARM, param1 = 1
            self.goto(State.WAIT_ARM)

        elif self.state is State.WAIT_ARM:
            armed = VehicleStatus.ARMING_STATE_ARMED
            if self.status and self.status.arming_state == armed:
                self.goto(State.HOVER)
            elif self.elapsed() > 5.0:
                self.get_logger().error("arming failed")
                self.goto(State.LAND)

        elif self.state is State.HOVER:
            self.setpoint = [0.0, 0.0, HOVER_Z]
            if self.elapsed() > HOVER_SECONDS:
                self.goto(State.LAND)

        elif self.state is State.LAND:
            self.publish_command(21)  # MAV_CMD_NAV_LAND
            self.goto(State.COMPLETE)

        elif self.state is State.COMPLETE:
            if self.status and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().info("mission complete, drone disarmed")
                self.destroy_node()
                rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardDemo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
