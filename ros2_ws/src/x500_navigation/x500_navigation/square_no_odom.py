#!/usr/bin/env python3
"""Square trajectory for the X500 without any odometry source.

The drone has no pose feedback: position is obtained by dead reckoning from the
IMU linear acceleration, and the yaw error comes from the IMU orientation. Each
side of the square is closed by a cascade of two PID loops:

    * distance loop  : keeps a constant forward speed until the side length is reached
    * yaw loop       : tracks the heading of the current side

The 2-D LiDAR is used only for the safety stop (obstacle closer than SAFETY_DIST).

Topics
    in   /x500/imu    sensor_msgs/Imu
    in   /x500/scan   sensor_msgs/LaserScan
    out  /x500/cmd_vel geometry_msgs/Twist
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

SIDE_LENGTH = 2.0        # m
CRUISE_SPEED = 0.35      # m/s
YAW_TOL = 0.12           # rad
DIST_TOL = 0.08          # m
SAFETY_DIST = 0.45       # m
CONTROL_DT = 0.02        # s


class Pid:
    def __init__(self, kp: float, ki: float, kd: float, limit: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def __call__(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = 0.0
        if self.prev_error is not None and dt > 0.0:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(max(-self.limit, min(self.limit, out)))


class X500SquareNoOdom(Node):
    def __init__(self) -> None:
        super().__init__("x500_square_no_odom")

        self.declare_parameter("side", SIDE_LENGTH)
        self.declare_parameter("speed", CRUISE_SPEED)
        self.declare_parameter("csv_path", "")

        self.side = float(self.get_parameter("side").value)
        self.speed = float(self.get_parameter("speed").value)
        self.csv_path = str(self.get_parameter("csv_path").value)

        self.cmd_pub = self.create_publisher(Twist, "/x500/cmd_vel", 10)
        self.create_subscription(Imu, "/x500/imu", self.on_imu, 10)
        self.create_subscription(LaserScan, "/x500/scan", self.on_scan, 10)

        # Dead-reckoning state
        self.vx = 0.0
        self.vy = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.accel = np.zeros(3)
        self.yaw_rate = 0.0
        self.last_imu_time: float | None = None
        self.min_scan = float("inf")

        # Controllers
        self.dist_pid = Pid(1.6, 0.05, 0.02, limit=self.speed)
        self.yaw_pid = Pid(1.8, 0.0, 0.15, limit=1.2)

        # Mission: four sides, each rotated by +90 deg
        self.side_index = 0
        self.side_start = (0.0, 0.0)
        self.done = False

        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t", "x_est", "y_est", "yaw", "side", "v_cmd", "w_cmd"])

        self.start_time = time.time()
        self.timer = self.create_timer(CONTROL_DT, self.control_step)
        self.get_logger().info(
            f"Square mission: side={self.side:.2f} m, cruise={self.speed:.2f} m/s"
        )

    # ------------------------------------------------------------------ callbacks
    def on_imu(self, msg: Imu) -> None:
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.last_imu_time is None else max(0.0, now - self.last_imu_time)
        self.last_imu_time = now
        if dt <= 0.0 or dt > 0.2:
            return

        # Orientation -> yaw (quaternion, only the yaw component is needed here)
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        # Linear acceleration is expressed in the body frame -> rotate to world
        ax, ay = msg.linear_acceleration.x, msg.linear_acceleration.y
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.accel = np.array([cy * ax - sy * ay, sy * ax + cy * ay, 0.0])
        self.yaw_rate = msg.angular_velocity.z

        # Dead reckoning (trapezoidal integration)
        self.vx += self.accel[0] * dt
        self.vy += self.accel[1] * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def on_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=float)
        valid = ranges[np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)]
        self.min_scan = float(valid.min()) if valid.size else float("inf")

    # ------------------------------------------------------------------ mission
    def target_of_side(self) -> tuple[float, float]:
        angle = math.pi / 2.0 * self.side_index
        return (
            self.side_start[0] + self.side * math.cos(angle),
            self.side_start[1] + self.side * math.sin(angle),
        )

    def control_step(self) -> None:
        cmd = Twist()
        if self.done:
            self.cmd_pub.publish(cmd)
            return

        tx, ty = self.target_of_side()
        dx, dy = tx - self.x, ty - self.y
        distance = math.hypot(dx, dy)
        heading_target = math.atan2(dy, dx)
        yaw_error = math.atan2(
            math.sin(heading_target - self.yaw), math.cos(heading_target - self.yaw)
        )

        v_cmd = self.dist_pid(distance, CONTROL_DT)
        w_cmd = self.yaw_pid(yaw_error, CONTROL_DT)

        # Do not translate while the heading is far off
        if abs(yaw_error) > YAW_TOL:
            v_cmd = 0.0
        # Safety stop
        if self.min_scan < SAFETY_DIST:
            v_cmd = 0.0
            self.get_logger().warn(f"Obstacle at {self.min_scan:.2f} m — stop", once=False)

        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)

        if self.csv_writer is not None:
            self.csv_writer.writerow([
                round(time.time() - self.start_time, 3),
                round(self.x, 4), round(self.y, 4), round(self.yaw, 4),
                self.side_index, round(v_cmd, 3), round(w_cmd, 3),
            ])

        if distance < DIST_TOL:
            self.get_logger().info(
                f"Side {self.side_index + 1}/4 closed at "
                f"({self.x:.2f}, {self.y:.2f})"
            )
            self.side_start = (self.x, self.y)
            self.dist_pid.reset()
            self.yaw_pid.reset()
            self.side_index += 1
            if self.side_index >= 4:
                self.done = True
                self.cmd_pub.publish(Twist())
                self.get_logger().info("Square completed — landing / idle")
                if self.csv_file is not None:
                    self.csv_file.close()

    def destroy_node(self) -> bool:
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = X500SquareNoOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
