from __future__ import annotations

import math
import random
from typing import List

from fusion_interfaces.msg import CameraConeDetection, CameraConeDetections
import rclpy
from rclpy.node import Node


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SyntheticCameraPublisher(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_camera_publisher")

        defaults = [
            ("output_topic", "/camera/cone_detections"),
            ("frame_id", "zedx_left_camera_optical_frame"),
            ("publish_rate_hz", 15.0),
            ("cone_count", 8),
            ("image_width", 1280),
            ("image_height", 720),
            ("dropout_probability", 0.02),
            ("oosm_probability", 0.05),
            ("max_oosm_delay_ms", 120.0),
            ("timestamp_jitter_std_ms", 4.0),
            ("confidence_mean", 0.85),
            ("confidence_std", 0.12),
            ("image_quality_base", 0.95),
            ("image_quality_noise", 0.03),
            ("random_seed", 7),
        ]
        for key, value in defaults:
            self.declare_parameter(key, value)

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.cone_count = int(self.get_parameter("cone_count").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.dropout_probability = float(self.get_parameter("dropout_probability").value)
        self.oosm_probability = float(self.get_parameter("oosm_probability").value)
        self.max_oosm_delay_ms = float(self.get_parameter("max_oosm_delay_ms").value)
        self.timestamp_jitter_std_ms = float(
            self.get_parameter("timestamp_jitter_std_ms").value
        )
        self.confidence_mean = float(self.get_parameter("confidence_mean").value)
        self.confidence_std = float(self.get_parameter("confidence_std").value)
        self.image_quality_base = float(self.get_parameter("image_quality_base").value)
        self.image_quality_noise = float(self.get_parameter("image_quality_noise").value)
        self.rng = random.Random(int(self.get_parameter("random_seed").value))
        self.step_index = 0

        self.pub = self.create_publisher(CameraConeDetections, self.output_topic, 10)
        self.create_timer(1.0 / max(1.0, self.publish_rate_hz), self._tick)

        self.get_logger().info(
            f"synthetic_camera_publisher -> {self.output_topic} ({self.publish_rate_hz:.1f} Hz)"
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

        msg = CameraConeDetections()
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.header.frame_id = self.frame_id
        msg.image_quality = float(
            _clamp(
                self.image_quality_base + self.rng.gauss(0.0, self.image_quality_noise),
                0.0,
                1.0,
            )
        )

        detections: List[CameraConeDetection] = []
        for i in range(max(1, self.cone_count)):
            det = CameraConeDetection()
            phase = 0.25 * self.step_index + 0.8 * i
            u = 0.5 + 0.42 * math.sin(phase)
            v = 0.62 - 0.22 * math.cos(0.7 * phase)
            box_w = _clamp(50.0 + 25.0 * self.rng.random(), 20.0, 120.0)
            box_h = _clamp(55.0 + 30.0 * self.rng.random(), 20.0, 140.0)

            cx = _clamp(u * self.image_width, 0.0, float(self.image_width))
            cy = _clamp(v * self.image_height, 0.0, float(self.image_height))
            det.xmin = float(_clamp(cx - 0.5 * box_w, 0.0, float(self.image_width)))
            det.ymin = float(_clamp(cy - 0.5 * box_h, 0.0, float(self.image_height)))
            det.xmax = float(_clamp(cx + 0.5 * box_w, 0.0, float(self.image_width)))
            det.ymax = float(_clamp(cy + 0.5 * box_h, 0.0, float(self.image_height)))

            conf = _clamp(
                self.confidence_mean + self.rng.gauss(0.0, self.confidence_std), 0.01, 1.0
            )
            det.confidence = float(conf)

            class_probs = [0.0, 0.0, 0.0, 0.0]
            dominant = i % 3  # cycle blue/yellow/orange
            class_probs[dominant] = 0.75 + 0.2 * self.rng.random()
            class_probs[3] = max(0.0, 1.0 - class_probs[dominant] - 0.05 * self.rng.random())
            rem = max(0.0, 1.0 - class_probs[dominant] - class_probs[3])
            class_probs[(dominant + 1) % 3] = 0.5 * rem
            class_probs[(dominant + 2) % 3] = 0.5 * rem
            norm = sum(class_probs)
            det.class_probs = [float(v / norm) for v in class_probs]
            detections.append(det)

        msg.detections = detections
        self.pub.publish(msg)
        self.step_index += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
