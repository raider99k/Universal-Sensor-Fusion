from __future__ import annotations

import math
import random
import struct
from typing import List, Optional, Tuple

try:
    from geometry_msgs.msg import Point, Pose, Pose2D, Quaternion, TransformStamped, Twist, Vector3
    from nav_msgs.msg import Odometry
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_ros import TransformBroadcaster
    from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesis, ObjectHypothesisWithPose
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    class Node:  # type: ignore
        pass
    class Point: pass
    class Pose2D: pass
    class Quaternion: pass
    class TransformStamped: pass
    class Odometry: pass
    class PointCloud2: pass
    class PointField: pass
    class Header: pass
    class TransformBroadcaster:
        def __init__(self, node): pass
    class Detection2DArray: pass
    class MarkerArray: pass


def _clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class ConeLandmark:
    def __init__(self, x: float, y: float, z: float, cone_class: str) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.cone_class = str(cone_class)  # "blue", "yellow", "orange"


class TrackModel:
    """
    Deterministic Formula Student circuit track generator.
    Produces static 3D cone landmarks in odom frame:
    - Left boundary: Blue cones (class 0)
    - Right boundary: Yellow cones (class 1)
    - Start/finish line: Orange cones (class 2)
    """

    def __init__(
        self,
        straight_length_m: float = 40.0,
        turn_radius_m: float = 15.0,
        track_width_m: float = 3.5,
        cone_spacing_m: float = 2.5,
    ) -> None:
        self.straight_len = float(straight_length_m)
        self.radius = float(turn_radius_m)
        self.width = float(track_width_m)
        self.spacing = float(cone_spacing_m)

        # 4 segments: Straight 1 (bottom), Turn 1 (right), Straight 2 (top), Turn 2 (left)
        self.arc_len = math.pi * self.radius
        self.total_length = 2.0 * self.straight_len + 2.0 * self.arc_len
        self.landmarks: List[ConeLandmark] = self._generate_cones()

    def get_pose_at_distance(self, s: float) -> Tuple[float, float, float, float]:
        """Returns (x, y, yaw, curvature) along centerline at arc length s."""
        s = s % self.total_length

        if s < self.straight_len:
            # Straight 1: from (-straight_len/2, -radius) to (straight_len/2, -radius)
            t = s / self.straight_len
            x = -0.5 * self.straight_len + t * self.straight_len
            y = -self.radius
            yaw = 0.0
            curv = 0.0
        elif s < self.straight_len + self.arc_len:
            # Turn 1: arc around (straight_len/2, 0) from -pi/2 to +pi/2
            ds = s - self.straight_len
            angle = -0.5 * math.pi + (ds / self.radius)
            x = 0.5 * self.straight_len + self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            yaw = angle + 0.5 * math.pi
            curv = 1.0 / self.radius
        elif s < 2.0 * self.straight_len + self.arc_len:
            # Straight 2: from (straight_len/2, radius) to (-straight_len/2, radius)
            ds = s - (self.straight_len + self.arc_len)
            t = ds / self.straight_len
            x = 0.5 * self.straight_len - t * self.straight_len
            y = self.radius
            yaw = math.pi
            curv = 0.0
        else:
            # Turn 2: arc around (-straight_len/2, 0) from +pi/2 to +3pi/2
            ds = s - (2.0 * self.straight_len + self.arc_len)
            angle = 0.5 * math.pi + (ds / self.radius)
            x = -0.5 * self.straight_len + self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            yaw = angle + 0.5 * math.pi
            curv = 1.0 / self.radius

        return x, y, yaw, curv

    def _generate_cones(self) -> List[ConeLandmark]:
        cones: List[ConeLandmark] = []
        num_steps = int(math.ceil(self.total_length / self.spacing))
        half_w = 0.5 * self.width

        for i in range(num_steps):
            s = i * self.spacing
            x, y, yaw, _ = self.get_pose_at_distance(s)

            # Normal vector pointing left
            nx = -math.sin(yaw)
            ny = math.cos(yaw)

            # Start/finish zone at s near 0: orange cones
            if s <= 2.5 or s >= self.total_length - 2.5:
                cones.append(ConeLandmark(x + half_w * nx, y + half_w * ny, 0.0, "orange"))
                cones.append(ConeLandmark(x - half_w * nx, y - half_w * ny, 0.0, "orange"))
            else:
                # Left boundary: blue; Right boundary: yellow
                cones.append(ConeLandmark(x + half_w * nx, y + half_w * ny, 0.0, "blue"))
                cones.append(ConeLandmark(x - half_w * nx, y - half_w * ny, 0.0, "yellow"))

        return cones


class SyntheticTrackPublisher(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_track_publisher")

        self._declare_parameters()
        self._load_parameters()

        self.track = TrackModel(
            straight_length_m=self.straight_length_m,
            turn_radius_m=self.turn_radius_m,
            track_width_m=self.track_width_m,
            cone_spacing_m=self.cone_spacing_m,
        )

        self.rng = random.Random(self.random_seed)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers for standard producer topics
        self.camera_pub = self.create_publisher(
            Detection2DArray, self.camera_topic, 10
        )
        self.lidar_pub = self.create_publisher(
            PointCloud2, self.lidar_topic, 10
        )
        self.motion_pub = self.create_publisher(
            Odometry, self.motion_topic, 20
        )
        self.ground_truth_pub = self.create_publisher(
            MarkerArray, self.ground_truth_topic, 1
        )

        # Vehicle state along track
        self.current_s = 0.0
        self.last_motion_time = self.get_clock().now()

        # Timers
        motion_period = 1.0 / max(1.0, self.motion_publish_rate_hz)
        self.motion_timer = self.create_timer(motion_period, self._on_motion_tick)

        camera_period = 1.0 / max(1.0, self.camera_publish_rate_hz)
        self.camera_timer = self.create_timer(camera_period, self._on_camera_tick)

        lidar_period = 1.0 / max(1.0, self.lidar_publish_rate_hz)
        self.lidar_timer = self.create_timer(lidar_period, self._on_lidar_tick)

        gt_period = 1.0 / max(0.5, self.ground_truth_publish_rate_hz)
        self.gt_timer = self.create_timer(gt_period, self._on_ground_truth_tick)

        # Prepare PointCloud2 fields layout
        self.lidar_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="t", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        self.lidar_point_step = 20

        self.get_logger().info(
            f"synthetic_track_publisher initialized: track length={self.track.total_length:.1f}m, "
            f"cones={len(self.track.landmarks)}, speed={self.vehicle_speed_mps:.1f} m/s, "
            f"camera={self.camera_publish_rate_hz:.1f}Hz, lidar={self.lidar_publish_rate_hz:.1f}Hz, "
            f"motion={self.motion_publish_rate_hz:.1f}Hz."
        )

    def _declare_parameters(self) -> None:
        defaults = [
            ("camera_topic", "/vision/cone_detections"),
            ("lidar_topic", "/ouster/points"),
            ("motion_topic", "/ego/odometry"),
            ("ground_truth_topic", "/ground_truth/cones"),
            ("target_frame", "base_link"),
            ("world_frame", "odom"),
            ("straight_length_m", 40.0),
            ("turn_radius_m", 15.0),
            ("track_width_m", 3.5),
            ("cone_spacing_m", 2.5),
            ("vehicle_speed_mps", 8.0),
            ("camera_publish_rate_hz", 20.0),
            ("lidar_publish_rate_hz", 20.0),
            ("motion_publish_rate_hz", 100.0),
            ("ground_truth_publish_rate_hz", 2.0),
            ("camera_fx", 700.0),
            ("camera_fy", 700.0),
            ("camera_cx", 640.0),
            ("camera_cy", 360.0),
            ("image_width", 1280),
            ("image_height", 720),
            ("camera_dropout_prob", 0.02),
            ("lidar_dropout_prob", 0.02),
            ("camera_oosm_prob", 0.05),
            ("lidar_oosm_prob", 0.08),
            ("max_oosm_delay_ms", 90.0),
            ("timestamp_jitter_std_ms", 3.0),
            ("random_seed", 42),
        ]
        for name, value in defaults:
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.motion_topic = str(self.get_parameter("motion_topic").value)
        self.ground_truth_topic = str(self.get_parameter("ground_truth_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.straight_length_m = float(self.get_parameter("straight_length_m").value)
        self.turn_radius_m = float(self.get_parameter("turn_radius_m").value)
        self.track_width_m = float(self.get_parameter("track_width_m").value)
        self.cone_spacing_m = float(self.get_parameter("cone_spacing_m").value)
        self.vehicle_speed_mps = float(self.get_parameter("vehicle_speed_mps").value)
        self.camera_publish_rate_hz = float(self.get_parameter("camera_publish_rate_hz").value)
        self.lidar_publish_rate_hz = float(self.get_parameter("lidar_publish_rate_hz").value)
        self.motion_publish_rate_hz = float(self.get_parameter("motion_publish_rate_hz").value)
        self.ground_truth_publish_rate_hz = float(
            self.get_parameter("ground_truth_publish_rate_hz").value
        )
        self.camera_fx = float(self.get_parameter("camera_fx").value)
        self.camera_fy = float(self.get_parameter("camera_fy").value)
        self.camera_cx = float(self.get_parameter("camera_cx").value)
        self.camera_cy = float(self.get_parameter("camera_cy").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.camera_dropout_prob = float(self.get_parameter("camera_dropout_prob").value)
        self.lidar_dropout_prob = float(self.get_parameter("lidar_dropout_prob").value)
        self.camera_oosm_prob = float(self.get_parameter("camera_oosm_prob").value)
        self.lidar_oosm_prob = float(self.get_parameter("lidar_oosm_prob").value)
        self.max_oosm_delay_ms = float(self.get_parameter("max_oosm_delay_ms").value)
        self.timestamp_jitter_std_ms = float(self.get_parameter("timestamp_jitter_std_ms").value)
        self.random_seed = int(self.get_parameter("random_seed").value)

    def _get_stamped_header(self, frame_id: str, oosm_prob: float) -> Header:
        now = self.get_clock().now()
        now_ns = now.nanoseconds

        # Timestamp jitter
        jitter_ns = int(self.rng.gauss(0.0, self.timestamp_jitter_std_ms * 1e6))
        ts_ns = max(0, now_ns + jitter_ns)

        # OOSM injection
        if self.rng.random() < oosm_prob:
            oosm_ns = int(self.rng.uniform(20.0, self.max_oosm_delay_ms) * 1e6)
            ts_ns = max(0, ts_ns - oosm_ns)

        header = Header()
        header.stamp = Time(nanoseconds=ts_ns).to_msg()
        header.frame_id = frame_id
        return header

    def _transform_to_ego(
        self, cone: ConeLandmark, vx: float, vy: float, vyaw: float
    ) -> Tuple[float, float, float]:
        dx = cone.x - vx
        dy = cone.y - vy
        cos_y = math.cos(vyaw)
        sin_y = math.sin(vyaw)
        x_ego = dx * cos_y + dy * sin_y
        y_ego = -dx * sin_y + dy * cos_y
        z_ego = cone.z
        return x_ego, y_ego, z_ego

    def _on_motion_tick(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_motion_time).nanoseconds * 1e-9
        self.last_motion_time = now

        if dt > 0.0 and dt < 1.0:
            self.current_s = (self.current_s + self.vehicle_speed_mps * dt) % self.track.total_length

        vx, vy, vyaw, curv = self.track.get_pose_at_distance(self.current_s)
        omega = self.vehicle_speed_mps * curv

        # Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = self.target_frame

        odom_msg.pose.pose.position.x = vx
        odom_msg.pose.pose.position.y = vy
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = _yaw_to_quaternion(vyaw)

        odom_msg.twist.twist.linear.x = self.vehicle_speed_mps
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = omega

        self.motion_pub.publish(odom_msg)

        # Broadcast TF odom -> base_link
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = now.to_msg()
        tf_stamped.header.frame_id = self.world_frame
        tf_stamped.child_frame_id = self.target_frame
        tf_stamped.transform.translation.x = vx
        tf_stamped.transform.translation.y = vy
        tf_stamped.transform.translation.z = 0.0
        tf_stamped.transform.rotation = _yaw_to_quaternion(vyaw)
        self.tf_broadcaster.sendTransform(tf_stamped)

    def _on_camera_tick(self) -> None:
        if self.rng.random() < self.camera_dropout_prob:
            return

        vx, vy, vyaw, _ = self.track.get_pose_at_distance(self.current_s)
        header = self._get_stamped_header("camera_optical_frame", self.camera_oosm_prob)

        det_array = Detection2DArray()
        det_array.header = header

        for cone in self.track.landmarks:
            x_ego, y_ego, z_ego = self._transform_to_ego(cone, vx, vy, vyaw)

            # Camera view frustum: in front, within depth range [1.8, 30.0]m
            if x_ego < 1.8 or x_ego > 30.0 or abs(y_ego) > 12.0:
                continue

            # Cone geometric center is at z = z_ego + 0.16m (cone height is 0.32m)
            cone_center_z = z_ego + 0.16
            x_cam = -y_ego
            y_cam = -cone_center_z
            z_cam = x_ego

            u = self.camera_fx * (x_cam / z_cam) + self.camera_cx
            v = self.camera_fy * (y_cam / z_cam) + self.camera_cy

            # Cone dimensions: height 0.32m, width 0.23m
            bbox_h = self.camera_fy * (0.32 / z_cam)
            bbox_w = self.camera_fx * (0.23 / z_cam)

            # Check inside image boundaries
            if (
                u + 0.5 * bbox_w < 0
                or u - 0.5 * bbox_w >= self.image_width
                or v + 0.5 * bbox_h < 0
                or v - 0.5 * bbox_h >= self.image_height
            ):
                continue

            # Add small noise to pixel coordinates
            u += self.rng.gauss(0.0, 1.5)
            v += self.rng.gauss(0.0, 1.5)
            bbox_w *= 1.0 + self.rng.gauss(0.0, 0.04)
            bbox_h *= 1.0 + self.rng.gauss(0.0, 0.04)

            detection = Detection2D()
            detection.header = header

            # Setup BoundingBox2D
            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.x = float(u)
            bbox.center.y = float(v)
            bbox.center.theta = 0.0
            bbox.size_x = float(bbox_w)
            bbox.size_y = float(bbox_h)
            detection.bbox = bbox

            # Setup hypothesis
            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis = ObjectHypothesis()
            hypo.hypothesis.class_id = cone.cone_class
            hypo.hypothesis.score = float(_clamp(self.rng.gauss(0.92, 0.04), 0.70, 0.99))
            detection.results.append(hypo)

            det_array.detections.append(detection)

        self.camera_pub.publish(det_array)

    def _on_lidar_tick(self) -> None:
        if self.rng.random() < self.lidar_dropout_prob:
            return

        vx, vy, vyaw, _ = self.track.get_pose_at_distance(self.current_s)
        header = self._get_stamped_header("ouster_lidar", self.lidar_oosm_prob)

        point_buffer = bytearray()
        point_count = 0

        # Heights to sample on the cone surface mantle
        height_levels = [0.06, 0.12, 0.18, 0.25]
        cone_base_radius = 0.12
        cone_height = 0.32

        for cone in self.track.landmarks:
            x_ego, y_ego, z_ego = self._transform_to_ego(cone, vx, vy, vyaw)

            # Range checks: must be past vehicle self-return radius (1.7m) and within ROI
            dist = math.hypot(x_ego, y_ego)
            if x_ego < 1.8 or dist > 30.0 or abs(y_ego) > 12.0:
                continue

            # Angles facing the vehicle
            bearing_to_vehicle = math.atan2(-y_ego, -x_ego)

            for h in height_levels:
                r_h = cone_base_radius * (1.0 - h / cone_height)
                # 4 angular samples facing the car
                for phi_offset in [-0.6, -0.2, 0.2, 0.6]:
                    phi = bearing_to_vehicle + phi_offset
                    px = x_ego + r_h * math.cos(phi) + self.rng.gauss(0.0, 0.015)
                    py = y_ego + r_h * math.sin(phi) + self.rng.gauss(0.0, 0.015)
                    pz = z_ego + h + self.rng.gauss(0.0, 0.015)

                    intensity = float(self.rng.gauss(180.0, 20.0))
                    t_scan = float(self.rng.uniform(0.0, 0.05))

                    point_buffer.extend(struct.pack("<fffff", px, py, pz, intensity, t_scan))
                    point_count += 1

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = point_count
        cloud_msg.fields = self.lidar_fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = self.lidar_point_step
        cloud_msg.row_step = self.lidar_point_step * point_count
        cloud_msg.data = bytes(point_buffer)
        cloud_msg.is_dense = True

        self.lidar_pub.publish(cloud_msg)

    def _on_ground_truth_tick(self) -> None:
        markers = MarkerArray()
        now = self.get_clock().now()

        for idx, cone in enumerate(self.track.landmarks):
            m = Marker()
            m.header.stamp = now.to_msg()
            m.header.frame_id = self.world_frame
            m.ns = "ground_truth_cones"
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = cone.x
            m.pose.position.y = cone.y
            m.pose.position.z = 0.16  # Center of 0.32m cone
            m.pose.orientation.w = 1.0

            m.scale.x = 0.24
            m.scale.y = 0.24
            m.scale.z = 0.32

            if cone.cone_class == "blue":
                m.color.r = 0.0
                m.color.g = 0.3
                m.color.b = 1.0
                m.color.a = 0.85
            elif cone.cone_class == "yellow":
                m.color.r = 1.0
                m.color.g = 0.9
                m.color.b = 0.0
                m.color.a = 0.85
            else:  # Orange
                m.color.r = 1.0
                m.color.g = 0.4
                m.color.b = 0.0
                m.color.a = 0.95

            markers.markers.append(m)

        self.ground_truth_pub.publish(markers)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SyntheticTrackPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
