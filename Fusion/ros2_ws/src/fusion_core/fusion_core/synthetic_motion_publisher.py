from __future__ import annotations

import math

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


def _yaw_to_quaternion(yaw: float):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class SyntheticMotionPublisher(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_motion_publisher")

        defaults = [
            ("output_topic", "/ego/odometry"),
            ("frame_id", "odom"),
            ("child_frame_id", "base_link"),
            ("publish_rate_hz", 100.0),
            ("speed_mps", 8.0),
            ("yaw_rate_rps", 0.08),
            ("radius_m", 40.0),
        ]
        for key, value in defaults:
            self.declare_parameter(key, value)

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.yaw_rate_rps = float(self.get_parameter("yaw_rate_rps").value)
        self.radius_m = float(self.get_parameter("radius_m").value)
        self.start_time_ns = self.get_clock().now().nanoseconds

        self.pub = self.create_publisher(Odometry, self.output_topic, 20)
        self.create_timer(1.0 / max(1.0, self.publish_rate_hz), self._tick)
        self.get_logger().info(
            f"synthetic_motion_publisher -> {self.output_topic} ({self.publish_rate_hz:.1f} Hz)"
        )

    def _tick(self) -> None:
        now = self.get_clock().now()
        t = max(0.0, (now.nanoseconds - self.start_time_ns) * 1e-9)

        yaw = self.yaw_rate_rps * t
        x = self.radius_m * math.sin(yaw)
        y = self.radius_m * (1.0 - math.cos(yaw))
        qx, qy, qz, qw = _yaw_to_quaternion(yaw)

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        msg.twist.twist.linear.x = float(self.speed_mps)
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = float(self.yaw_rate_rps)

        msg.pose.covariance = [
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.02,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.02,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.02,
        ]
        msg.twist.covariance = msg.pose.covariance
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticMotionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
