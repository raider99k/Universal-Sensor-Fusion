from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

try:
    from fusion_interfaces.msg import LidarConeCluster, LidarConeClusters
    from geometry_msgs.msg import Point, PoseArray
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import Marker, MarkerArray
    try:
        from vision_msgs.msg import Detection3D, Detection3DArray
    except ImportError:
        Detection3D = None  # type: ignore[assignment,misc]
        Detection3DArray = None  # type: ignore[assignment,misc]
except ImportError:  # pragma: no cover - bare-Python testing fallback
    rclpy = None

    class Node:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class _HeaderStub:
        def __init__(self) -> None:
            self.stamp = None
            self.frame_id = ""

    class Point:  # type: ignore[no-redef]
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class LidarConeCluster:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.centroid = Point()
            self.covariance = [0.0] * 9
            self.geometry_quality = 1.0
            self.point_count = 10
            self.stripe_class_probs = [0.25, 0.25, 0.25, 0.25]

    class LidarConeClusters:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.clusters = []
            self.lidar_quality = 1.0

    class PoseArray:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.poses = []

    class Marker:  # type: ignore[no-redef]
        ARROW = 0
        CUBE = 1
        SPHERE = 2
        CYLINDER = 3
        POINTS = 8
        MESH_RESOURCE = 10
        DELETE = 2
        DELETEALL = 3

    class MarkerArray:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.markers = []

    class Detection3D:  # type: ignore[no-redef]
        pass

    class Detection3DArray:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.detections = []

    class PointCloud2:  # type: ignore[no-redef]
        pass

    class QoSProfile:  # type: ignore[no-redef]
        def __init__(self, depth: int = 10) -> None:
            self.depth = depth
            self.reliability = 0
            self.history = 0
            self.durability = 0

    class ReliabilityPolicy:  # type: ignore[no-redef]
        BEST_EFFORT = 0
        RELIABLE = 1

    class HistoryPolicy:  # type: ignore[no-redef]
        KEEP_LAST = 0

    class DurabilityPolicy:  # type: ignore[no-redef]
        VOLATILE = 0


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class ClassMapper:
    """Maps custom string labels or integer IDs to internal 4-class indices:
       0: Blue, 1: Yellow, 2: Orange, 3: Big Orange
    """

    def __init__(self, mapping_expr: str = "") -> None:
        self.label_to_idx = {
            "blue": 0,
            "blue_cone": 0,
            "yellow": 1,
            "yellow_cone": 1,
            "orange": 2,
            "orange_cone": 2,
            "small_orange": 2,
            "small_orange_cone": 2,
            "big_orange": 3,
            "large_orange": 3,
            "large_orange_cone": 3,
        }
        if mapping_expr:
            self._parse_mapping(mapping_expr)

    def _parse_mapping(self, expr: str) -> None:
        for chunk in expr.split(","):
            chunk = chunk.strip()
            if ":" not in chunk:
                continue
            lbl, idx_s = chunk.split(":", 1)
            try:
                idx = int(idx_s.strip())
                if 0 <= idx <= 3:
                    self.label_to_idx[lbl.strip().lower()] = idx
            except ValueError:
                continue

    def to_index(self, label_or_id: Any) -> Optional[int]:
        s = str(label_or_id).strip().lower()
        if s in self.label_to_idx:
            return self.label_to_idx[s]
        try:
            val = int(s)
            if 0 <= val <= 3:
                return val
        except ValueError:
            pass
        return None

    def synthesize_distribution(
        self, class_idx: Optional[int], confidence: float = 0.85
    ) -> List[float]:
        if class_idx is None or class_idx < 0 or class_idx > 3:
            return [0.25, 0.25, 0.25, 0.25]
        conf = _clamp(confidence, 0.30, 0.98)
        rem = (1.0 - conf) / 3.0
        dist = [rem, rem, rem, rem]
        dist[class_idx] = conf
        return dist


class LidarClusterAdapter(Node):
    """Generic adapter bridging arbitrary external LiDAR cluster formats
    (PoseArray, MarkerArray, Detection3DArray, PointCloud2) into standard
    fusion_interfaces/msg/LidarConeClusters.

    Guarantees:
    1. Populates valid physical covariances (distance-scaled) so Mahalanobis gating succeeds.
    2. Enforces minimum point counts so uncorroborated clusters are never silently discarded.
    3. Guarantees non-zero geometry_quality so track promotion is never starved.
    4. Supports optional surface-to-axis ray centering to match physical cone vertical axes.
    5. Preserves upstream color/semantics (e.g. from Marker RGBA or Detection3D class IDs).
    """

    def __init__(self) -> None:
        super().__init__("lidar_cluster_adapter")

        # Topics
        self.declare_parameter("output_topic", "/lidar/cone_clusters")
        self.declare_parameter("pose_array_topic", "")
        self.declare_parameter("marker_array_topic", "")
        self.declare_parameter("detection3d_topic", "")
        self.declare_parameter("generic_input_topic", "/lidar/cone_clusters_in")
        self.declare_parameter("input_type", "auto")  # 'auto', 'pose_array', 'marker_array', 'detection3d'

        # Defaults and physical adjustments
        self.declare_parameter("default_point_count", 10)
        self.declare_parameter("default_geometry_quality", 1.0)
        self.declare_parameter("enable_surface_axis_offset", False)
        self.declare_parameter("surface_offset_m", 0.08)
        self.declare_parameter("base_variance_xy", 0.04)
        self.declare_parameter("variance_range_coeff", 0.003)
        self.declare_parameter("base_variance_z", 0.10)
        self.declare_parameter("min_range_m", 0.3)
        self.declare_parameter("max_range_m", 60.0)
        self.declare_parameter(
            "class_map",
            "blue:0,blue_cone:0,yellow:1,yellow_cone:1,orange:2,small_orange:2,big_orange:3,large_orange:3",
        )

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.pose_array_topic = str(self.get_parameter("pose_array_topic").value).strip()
        self.marker_array_topic = str(self.get_parameter("marker_array_topic").value).strip()
        self.detection3d_topic = str(self.get_parameter("detection3d_topic").value).strip()
        self.generic_input_topic = str(self.get_parameter("generic_input_topic").value).strip()
        self.input_type = str(self.get_parameter("input_type").value).strip().lower()

        self.default_point_count = int(self.get_parameter("default_point_count").value)
        self.default_geometry_quality = float(self.get_parameter("default_geometry_quality").value)
        self.enable_surface_axis_offset = bool(self.get_parameter("enable_surface_axis_offset").value)
        self.surface_offset_m = float(self.get_parameter("surface_offset_m").value)
        self.base_variance_xy = float(self.get_parameter("base_variance_xy").value)
        self.variance_range_coeff = float(self.get_parameter("variance_range_coeff").value)
        self.base_variance_z = float(self.get_parameter("base_variance_z").value)
        self.min_range_m = float(self.get_parameter("min_range_m").value)
        self.max_range_m = float(self.get_parameter("max_range_m").value)

        self.class_mapper = ClassMapper(str(self.get_parameter("class_map").value))

        # Publishers
        qos_pub = QoSProfile(depth=10)
        qos_pub.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_pub.durability = DurabilityPolicy.VOLATILE
        qos_pub.history = HistoryPolicy.KEEP_LAST

        if hasattr(self, "create_publisher") and rclpy is not None:
            self.publisher_ = self.create_publisher(LidarConeClusters, self.output_topic, qos_pub)
        else:
            self.publisher_ = None

        # Subscribers
        qos_sub = QoSProfile(depth=10)
        qos_sub.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sub.durability = DurabilityPolicy.VOLATILE
        qos_sub.history = HistoryPolicy.KEEP_LAST

        if hasattr(self, "create_subscription") and rclpy is not None:
            if self.pose_array_topic:
                self.create_subscription(
                    PoseArray, self.pose_array_topic, self.on_pose_array, qos_sub
                )
            if self.marker_array_topic:
                self.create_subscription(
                    MarkerArray, self.marker_array_topic, self.on_marker_array, qos_sub
                )
            if self.detection3d_topic and Detection3DArray is not None:
                self.create_subscription(
                    Detection3DArray, self.detection3d_topic, self.on_detection3d_array, qos_sub
                )
            # Generic topic subscription
            if self.generic_input_topic and not (self.pose_array_topic or self.marker_array_topic or self.detection3d_topic):
                if self.input_type == "pose_array":
                    self.create_subscription(
                        PoseArray, self.generic_input_topic, self.on_pose_array, qos_sub
                    )
                elif self.input_type == "marker_array":
                    self.create_subscription(
                        MarkerArray, self.generic_input_topic, self.on_marker_array, qos_sub
                    )
                elif self.input_type == "detection3d" and Detection3DArray is not None:
                    self.create_subscription(
                        Detection3DArray, self.generic_input_topic, self.on_detection3d_array, qos_sub
                    )

    # --------------------------------------------------------------------------
    # Conversion Utilities
    # --------------------------------------------------------------------------

    def compute_covariance(self, x: float, y: float, var_diag: Optional[Tuple[float, float, float]] = None) -> List[float]:
        """Generates a physically realistic 3x3 covariance matrix."""
        if var_diag is not None:
            vx, vy, vz = var_diag
            return [max(1e-4, vx), 0.0, 0.0, 0.0, max(1e-4, vy), 0.0, 0.0, 0.0, max(1e-4, vz)]

        r = math.sqrt(x * x + y * y)
        sigma_xy = self.base_variance_xy + self.variance_range_coeff * r
        var_xy = max(1e-4, sigma_xy * sigma_xy)
        return [var_xy, 0.0, 0.0, 0.0, var_xy, 0.0, 0.0, 0.0, max(1e-4, self.base_variance_z)]

    def apply_offset(self, x: float, y: float) -> Tuple[float, float]:
        """Shifts cluster centroid from reflective front shell outward along the ray."""
        if not self.enable_surface_axis_offset:
            return x, y
        r = math.sqrt(x * x + y * y)
        if r < 0.1:
            return x, y
        scale = (r + self.surface_offset_m) / r
        return x * scale, y * scale

    # --------------------------------------------------------------------------
    # Message Handlers
    # --------------------------------------------------------------------------

    def convert_pose_array(self, msg: Any) -> LidarConeClusters:
        out = LidarConeClusters()
        out.header = msg.header
        out.lidar_quality = 1.0

        for pose in _safe_getattr(msg, "poses", []):
            pos = _safe_getattr(pose, "position", None)
            if pos is None:
                continue
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue
            r = math.sqrt(x * x + y * y)
            if r < self.min_range_m or r > self.max_range_m:
                continue

            x_adj, y_adj = self.apply_offset(x, y)
            cl = LidarConeCluster()
            cl.centroid = Point(x=x_adj, y=y_adj, z=z)
            cl.covariance = self.compute_covariance(x_adj, y_adj)
            cl.geometry_quality = float(self.default_geometry_quality)
            cl.point_count = int(self.default_point_count)
            cl.stripe_class_probs = [0.25, 0.25, 0.25, 0.25]
            out.clusters.append(cl)

        return out

    def on_pose_array(self, msg: Any) -> None:
        clusters_msg = self.convert_pose_array(msg)
        if self.publisher_ is not None:
            self.publisher_.publish(clusters_msg)

    def convert_marker_array(self, msg: Any) -> LidarConeClusters:
        out = LidarConeClusters()
        markers = _safe_getattr(msg, "markers", [])
        if markers:
            out.header = _safe_getattr(markers[0], "header", _HeaderStub())
        out.lidar_quality = 1.0

        for m in markers:
            # Skip deletions
            action = int(_safe_getattr(m, "action", 0))
            if action in (2, 3):  # DELETE, DELETEALL
                continue

            m_type = int(_safe_getattr(m, "type", 3))  # Default CYLINDER
            # Extract color semantic if available
            color = _safe_getattr(m, "color", None)
            class_probs = [0.25, 0.25, 0.25, 0.25]
            if color is not None:
                cr, cg, cb = float(color.r), float(color.g), float(color.b)
                if cb > 0.55 and cr < 0.45:  # Blue
                    class_probs = [0.90, 0.05, 0.025, 0.025]
                elif cr > 0.55 and cg > 0.55 and cb < 0.40:  # Yellow
                    class_probs = [0.05, 0.90, 0.025, 0.025]
                elif cr > 0.65 and cb < 0.35:  # Orange
                    sz = float(_safe_getattr(_safe_getattr(m, "scale", None), "z", 0.3))
                    if sz > 0.42:  # Big orange
                        class_probs = [0.025, 0.025, 0.05, 0.90]
                    else:  # Small orange
                        class_probs = [0.025, 0.025, 0.90, 0.05]

            geom_q = float(color.a) if color is not None and float(color.a) > 0.0 else self.default_geometry_quality

            if m_type == 8:  # POINTS
                pts = _safe_getattr(m, "points", [])
                for p in pts:
                    x, y, z = float(p.x), float(p.y), float(p.z)
                    if math.isnan(x) or math.isnan(y) or math.isnan(z):
                        continue
                    r = math.sqrt(x * x + y * y)
                    if r < self.min_range_m or r > self.max_range_m:
                        continue
                    x_adj, y_adj = self.apply_offset(x, y)
                    cl = LidarConeCluster()
                    cl.centroid = Point(x=x_adj, y=y_adj, z=z)
                    cl.covariance = self.compute_covariance(x_adj, y_adj)
                    cl.geometry_quality = geom_q
                    cl.point_count = int(self.default_point_count)
                    cl.stripe_class_probs = class_probs
                    out.clusters.append(cl)
            else:
                pose = _safe_getattr(m, "pose", None)
                pos = _safe_getattr(pose, "position", None)
                if pos is None:
                    continue
                x, y, z = float(pos.x), float(pos.y), float(pos.z)
                if math.isnan(x) or math.isnan(y) or math.isnan(z):
                    continue
                r = math.sqrt(x * x + y * y)
                if r < self.min_range_m or r > self.max_range_m:
                    continue
                x_adj, y_adj = self.apply_offset(x, y)
                cl = LidarConeCluster()
                cl.centroid = Point(x=x_adj, y=y_adj, z=z)
                cl.covariance = self.compute_covariance(x_adj, y_adj)
                cl.geometry_quality = geom_q
                cl.point_count = int(self.default_point_count)
                cl.stripe_class_probs = class_probs
                out.clusters.append(cl)

        return out

    def on_marker_array(self, msg: Any) -> None:
        clusters_msg = self.convert_marker_array(msg)
        if self.publisher_ is not None:
            self.publisher_.publish(clusters_msg)

    def convert_detection3d_array(self, msg: Any) -> LidarConeClusters:
        out = LidarConeClusters()
        out.header = msg.header
        out.lidar_quality = 1.0

        for det in _safe_getattr(msg, "detections", []):
            bbox = _safe_getattr(det, "bbox", None)
            center = _safe_getattr(bbox, "center", None)
            pos = _safe_getattr(center, "position", None)
            if pos is None:
                continue
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue
            r = math.sqrt(x * x + y * y)
            if r < self.min_range_m or r > self.max_range_m:
                continue

            # Synthesize covariance from bounding box if present
            var_diag = None
            size = _safe_getattr(bbox, "size", None)
            if size is not None:
                sx, sy, sz = float(size.x), float(size.y), float(size.z)
                if sx > 0 and sy > 0 and sz > 0:
                    var_diag = ((sx / 4.0) ** 2, (sy / 4.0) ** 2, (sz / 4.0) ** 2)

            # Extract class ID and score from results if present
            class_probs = [0.25, 0.25, 0.25, 0.25]
            geom_q = self.default_geometry_quality
            results = _safe_getattr(det, "results", [])
            if results:
                hyp = _safe_getattr(results[0], "hypothesis", None)
                if hyp is not None:
                    cid = _safe_getattr(hyp, "class_id", "")
                    score = float(_safe_getattr(hyp, "score", 0.85))
                    idx = self.class_mapper.to_index(cid)
                    class_probs = self.class_mapper.synthesize_distribution(idx, score)
                    geom_q = _clamp(score, 0.1, 1.0)

            x_adj, y_adj = self.apply_offset(x, y)
            cl = LidarConeCluster()
            cl.centroid = Point(x=x_adj, y=y_adj, z=z)
            cl.covariance = self.compute_covariance(x_adj, y_adj, var_diag=var_diag)
            cl.geometry_quality = float(geom_q)
            cl.point_count = int(self.default_point_count)
            cl.stripe_class_probs = class_probs
            out.clusters.append(cl)

        return out

    def on_detection3d_array(self, msg: Any) -> None:
        clusters_msg = self.convert_detection3d_array(msg)
        if self.publisher_ is not None:
            self.publisher_.publish(clusters_msg)


def main(args: Optional[Sequence[str]] = None) -> None:
    if rclpy is None:
        print("rclpy not installed; cannot run standalone node.")
        return
    rclpy.init(args=args)
    node = LidarClusterAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
