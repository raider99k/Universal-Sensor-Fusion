from __future__ import annotations

import math
import random
from typing import List

from fusion_interfaces.msg import LidarConeCluster, LidarConeClusters
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SyntheticLidarPublisher(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_lidar_publisher")

        defaults = [
            ("output_topic", "/lidar/cone_clusters"),
            ("frame_id", "base_link"),
            ("publish_rate_hz", 10.0),
            ("cone_count", 8),
            ("range_min_m", 4.0),
            ("range_max_m", 30.0),
            ("dropout_probability", 0.02),
            ("oosm_probability", 0.08),
            ("max_oosm_delay_ms", 160.0),
            ("timestamp_jitter_std_ms", 5.0),
            ("centroid_noise_xy_m", 0.06),
            ("centroid_noise_z_m", 0.03),
            ("lidar_quality_base", 0.93),
            ("lidar_quality_noise", 0.04),
            ("random_seed", 17),
        ]
        for key, value in defaults:
            self.declare_parameter(key, value)

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.cone_count = int(self.get_parameter("cone_count").value)
        self.range_min_m = float(self.get_parameter("range_min_m").value)
        self.range_max_m = float(self.get_parameter("range_max_m").value)
        self.dropout_probability = float(self.get_parameter("dropout_probability").value)
        self.oosm_probability = float(self.get_parameter("oosm_probability").value)
        self.max_oosm_delay_ms = float(self.get_parameter("max_oosm_delay_ms").value)
        self.timestamp_jitter_std_ms = float(
            self.get_parameter("timestamp_jitter_std_ms").value
        )
        self.centroid_noise_xy_m = float(self.get_parameter("centroid_noise_xy_m").value)
        self.centroid_noise_z_m = float(self.get_parameter("centroid_noise_z_m").value)
        self.lidar_quality_base = float(self.get_parameter("lidar_quality_base").value)
        self.lidar_quality_noise = float(self.get_parameter("lidar_quality_noise").value)
        self.rng = random.Random(int(self.get_parameter("random_seed").value))
        self.step_index = 0

        self.pub = self.create_publisher(LidarConeClusters, self.output_topic, 10)
        self.create_timer(1.0 / max(1.0, self.publish_rate_hz), self._tick)
        self.get_logger().info(
            f"synthetic_lidar_publisher -> {self.output_topic} ({self.publish_rate_hz:.1f} Hz)"
        )

    def _tick(self) -> None:
        if self.rng.random() < self.dropout_probability:
            return

        now_ns = self.get_clock().now().nanoseconds
        jitter_ns = int(self.rng.gauss(0.0, self.timestamp_jitter_std_ms) * 1e6)
        oosm_ns = 0
        if self.rng.random() < self.oosm_probability:
            oosm_ns = int(self.rng.uniform(0.0, self.max_oosm_delay_ms) * 1e6)
        stamp_ns = max(0, now_ns + jitter_ns - oosm_ns)

        msg = LidarConeClusters()
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.header.frame_id = self.frame_id
        msg.lidar_quality = float(
            _clamp(
                self.lidar_quality_base + self.rng.gauss(0.0, self.lidar_quality_noise),
                0.0,
                1.0,
            )
        )

        clusters: List[LidarConeCluster] = []
        for i in range(max(1, self.cone_count)):
            phase = 0.22 * self.step_index + 0.9 * i
            r = self.range_min_m + (self.range_max_m - self.range_min_m) * (
                0.2 + 0.75 * (0.5 + 0.5 * math.sin(0.2 * phase))
            )
            theta = -0.4 + 0.8 * (i / max(1, self.cone_count - 1))
            x = r * math.cos(theta) + self.rng.gauss(0.0, self.centroid_noise_xy_m)
            y = r * math.sin(theta) + self.rng.gauss(0.0, self.centroid_noise_xy_m)
            z = 0.18 + self.rng.gauss(0.0, self.centroid_noise_z_m)

            cluster = LidarConeCluster()
            cluster.centroid = Point(x=float(x), y=float(y), z=float(z))
            var_xy = _clamp(0.02 + 0.005 * (r / max(1.0, self.range_max_m)), 0.02, 0.08)
            var_z = _clamp(0.04 + 0.01 * self.rng.random(), 0.03, 0.1)
            cov_xy = 0.001 * self.rng.uniform(-1.0, 1.0)
            cluster.covariance = [
                float(var_xy),
                float(cov_xy),
                0.0,
                float(cov_xy),
                float(var_xy),
                0.0,
                0.0,
                0.0,
                float(var_z),
            ]
            cluster.geometry_quality = float(_clamp(0.75 + 0.2 * self.rng.random(), 0.0, 1.0))
            cluster.point_count = int(8 + 14 * self.rng.random())
            clusters.append(cluster)

        msg.clusters = clusters
        self.pub.publish(msg)
        self.step_index += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticLidarPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
