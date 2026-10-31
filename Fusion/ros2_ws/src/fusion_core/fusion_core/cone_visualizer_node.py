from __future__ import annotations

import math
from typing import List, Optional

try:
    from builtin_interfaces.msg import Duration
    from fusion_interfaces.msg import ConeObservation, ConeTrack
    from geometry_msgs.msg import Point, Vector3
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    class Node:  # type: ignore
        pass
    class Duration: pass
    class ConeTrack:
        TRACK_TENTATIVE = 0
        TRACK_CONFIRMED = 1
        TRACK_LOST = 2
    class ConeObservation: pass
    class Point: pass
    class Vector3: pass
    class Marker:
        CYLINDER = 3
        TEXT_VIEW_FACING = 9
        SPHERE = 2
        ADD = 0
        DELETE = 2
    class MarkerArray: pass

CLASS_LABELS = ["blue", "yellow", "orange", "unknown"]
CLASS_COLORS = {
    0: (0.0, 0.35, 1.0),    # Blue
    1: (1.0, 0.90, 0.0),    # Yellow
    2: (1.0, 0.40, 0.0),    # Orange
    3: (0.7, 0.70, 0.7),    # Unknown / Grey
}


class ConeVisualizerNode(Node):
    """
    Translates fusion contracts:
    - /perception/cone_tracks -> /perception/cone_track_markers
    - /perception/cone_observations -> /perception/cone_observation_markers
    Enables instant out-of-the-box 3D scene rendering in Foxglove Studio and RViz2.
    """

    def __init__(self) -> None:
        super().__init__("cone_visualizer_node")

        self.declare_parameter("tracks_topic", "/perception/cone_tracks")
        self.declare_parameter("observations_topic", "/perception/cone_observations")
        self.declare_parameter("track_markers_topic", "/perception/cone_track_markers")
        self.declare_parameter("observation_markers_topic", "/perception/cone_observation_markers")
        self.declare_parameter("track_lifetime_s", 0.30)
        self.declare_parameter("obs_lifetime_s", 0.20)

        self.tracks_topic = str(self.get_parameter("tracks_topic").value)
        self.observations_topic = str(self.get_parameter("observations_topic").value)
        self.track_markers_topic = str(self.get_parameter("track_markers_topic").value)
        self.observation_markers_topic = str(
            self.get_parameter("observation_markers_topic").value
        )
        self.track_lifetime_s = float(self.get_parameter("track_lifetime_s").value)
        self.obs_lifetime_s = float(self.get_parameter("obs_lifetime_s").value)

        self.track_sub = self.create_subscription(
            ConeTrack, self.tracks_topic, self._on_cone_track, 50
        )
        self.obs_sub = self.create_subscription(
            ConeObservation, self.observations_topic, self._on_cone_observation, 50
        )

        self.track_markers_pub = self.create_publisher(
            MarkerArray, self.track_markers_topic, 20
        )
        self.obs_markers_pub = self.create_publisher(
            MarkerArray, self.observation_markers_topic, 20
        )

        self.obs_counter = 0
        self.get_logger().info(
            f"cone_visualizer_node started: {self.tracks_topic} -> {self.track_markers_topic}, "
            f"{self.observations_topic} -> {self.observation_markers_topic}"
        )

    def _dominant_class(self, class_probs: List[float]) -> int:
        if not class_probs:
            return 3
        best_idx = 0
        best_val = class_probs[0]
        for i in range(1, len(class_probs)):
            if class_probs[i] > best_val:
                best_val = class_probs[i]
                best_idx = i
        if best_val < 0.35:
            return 3
        return best_idx

    def _on_cone_track(self, msg: ConeTrack) -> None:
        markers = MarkerArray()
        frame_id = msg.header.frame_id if msg.header.frame_id else "base_link"
        stamp = msg.header.stamp

        base_id = int(msg.track_id) * 10

        # Handle deleted / lost tracks
        if msg.track_status == ConeTrack.TRACK_LOST:
            for sub_id in range(4):
                del_marker = Marker()
                del_marker.header.frame_id = frame_id
                del_marker.header.stamp = stamp
                del_marker.ns = "cone_tracks"
                del_marker.id = base_id + sub_id
                del_marker.action = Marker.DELETE
                markers.markers.append(del_marker)
            self.track_markers_pub.publish(markers)
            return

        dominant_idx = self._dominant_class(msg.class_probs)
        rgb = CLASS_COLORS.get(dominant_idx, (0.7, 0.7, 0.7))
        alpha = 0.95 if msg.track_status == ConeTrack.TRACK_CONFIRMED else 0.60
        dominant_label = CLASS_LABELS[dominant_idx] if dominant_idx < len(CLASS_LABELS) else "unknown"

        lifetime = Duration()
        lifetime.sec = int(self.track_lifetime_s)
        lifetime.nanosec = int((self.track_lifetime_s - int(self.track_lifetime_s)) * 1e9)

        # 1. 3D Cone cylinder marker
        cone_m = Marker()
        cone_m.header.frame_id = frame_id
        cone_m.header.stamp = stamp
        cone_m.ns = "cone_tracks"
        cone_m.id = base_id + 0
        cone_m.type = Marker.CYLINDER
        cone_m.action = Marker.ADD
        cone_m.pose.position.x = msg.position.x
        cone_m.pose.position.y = msg.position.y
        cone_m.pose.position.z = msg.position.z + 0.16
        cone_m.pose.orientation.w = 1.0
        cone_m.scale.x = 0.25
        cone_m.scale.y = 0.25
        cone_m.scale.z = 0.32
        cone_m.color.r = rgb[0]
        cone_m.color.g = rgb[1]
        cone_m.color.b = rgb[2]
        cone_m.color.a = alpha
        cone_m.lifetime = lifetime
        markers.markers.append(cone_m)

        # 2. Floating track text marker (ID, label, quality)
        text_m = Marker()
        text_m.header.frame_id = frame_id
        text_m.header.stamp = stamp
        text_m.ns = "cone_tracks"
        text_m.id = base_id + 1
        text_m.type = Marker.TEXT_VIEW_FACING
        text_m.action = Marker.ADD
        text_m.pose.position.x = msg.position.x
        text_m.pose.position.y = msg.position.y
        text_m.pose.position.z = msg.position.z + 0.45
        text_m.pose.orientation.w = 1.0
        text_m.scale.z = 0.22
        text_m.text = f"ID:{msg.track_id} [{dominant_label.upper()}] {msg.quality_score:.2f}"
        text_m.color.r = 1.0
        text_m.color.g = 1.0
        text_m.color.b = 1.0
        text_m.color.a = 0.95
        text_m.lifetime = lifetime
        markers.markers.append(text_m)

        # 3. Covariance box/ellipsoid marker
        cov_m = Marker()
        cov_m.header.frame_id = frame_id
        cov_m.header.stamp = stamp
        cov_m.ns = "cone_tracks"
        cov_m.id = base_id + 2
        cov_m.type = Marker.SPHERE
        cov_m.action = Marker.ADD
        cov_m.pose.position.x = msg.position.x
        cov_m.pose.position.y = msg.position.y
        cov_m.pose.position.z = msg.position.z + 0.16
        cov_m.pose.orientation.w = 1.0

        var_x = msg.covariance[0] if len(msg.covariance) >= 9 else 0.05
        var_y = msg.covariance[4] if len(msg.covariance) >= 9 else 0.05
        var_z = msg.covariance[8] if len(msg.covariance) >= 9 else 0.05
        cov_m.scale.x = max(0.15, 2.0 * math.sqrt(max(0.001, float(var_x))))
        cov_m.scale.y = max(0.15, 2.0 * math.sqrt(max(0.001, float(var_y))))
        cov_m.scale.z = max(0.15, 2.0 * math.sqrt(max(0.001, float(var_z))))

        cov_m.color.r = rgb[0]
        cov_m.color.g = rgb[1]
        cov_m.color.b = rgb[2]
        cov_m.color.a = 0.25
        cov_m.lifetime = lifetime
        markers.markers.append(cov_m)

        self.track_markers_pub.publish(markers)

    def _on_cone_observation(self, msg: ConeObservation) -> None:
        markers = MarkerArray()
        frame_id = msg.header.frame_id if msg.header.frame_id else "base_link"
        stamp = msg.header.stamp

        self.obs_counter = (self.obs_counter + 1) % 500

        dominant_idx = self._dominant_class(msg.class_probs)
        rgb = CLASS_COLORS.get(dominant_idx, (0.7, 0.7, 0.7))

        lifetime = Duration()
        lifetime.sec = int(self.obs_lifetime_s)
        lifetime.nanosec = int((self.obs_lifetime_s - int(self.obs_lifetime_s)) * 1e9)

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "cone_observations"
        m.id = self.obs_counter
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = msg.position.x
        m.pose.position.y = msg.position.y
        m.pose.position.z = msg.position.z + 0.16
        m.pose.orientation.w = 1.0
        m.scale.x = 0.18
        m.scale.y = 0.18
        m.scale.z = 0.18
        m.color.r = rgb[0]
        m.color.g = rgb[1]
        m.color.b = rgb[2]
        m.color.a = 0.75
        m.lifetime = lifetime
        markers.markers.append(m)

        self.obs_markers_pub.publish(markers)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ConeVisualizerNode()
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
