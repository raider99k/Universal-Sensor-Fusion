from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from fusion_interfaces.msg import (
        CameraConeDetections,
        ConeObservation,
        ConeTrack,
        LidarConeCluster,
        LidarConeClusters,
    )
    from geometry_msgs.msg import (
        Point,
        PoseStamped,
        Quaternion,
        TwistStamped,
        TwistWithCovarianceStamped,
        Vector3,
    )
    from nav_msgs.msg import Odometry, Path
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import ColorRGBA, String
    from tf2_ros import Buffer, TransformException, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    rclpy = None  # type: ignore

    class Node:  # type: ignore
        pass

    class _HeaderStub:
        def __init__(self) -> None:
            self.stamp = None
            self.frame_id = ""

    class Point:  # type: ignore
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class Quaternion:  # type: ignore
        def __init__(
            self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0
        ) -> None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.w = float(w)

    class _PoseStub:
        def __init__(self) -> None:
            self.position = Point()
            self.orientation = Quaternion()

    class PoseStamped:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.pose = _PoseStub()

    class Path:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.poses: List[PoseStamped] = []

    class ColorRGBA:  # type: ignore
        def __init__(
            self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0
        ) -> None:
            self.r = float(r)
            self.g = float(g)
            self.b = float(b)
            self.a = float(a)

    class String:  # type: ignore
        def __init__(self, data: str = "") -> None:
            self.data = str(data)

    class _ScaleStub:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Marker:  # type: ignore
        ARROW = 0
        CUBE = 1
        SPHERE = 2
        CYLINDER = 3
        LINE_STRIP = 4
        TEXT_VIEW_FACING = 9
        ADD = 0
        DELETE = 2

        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.ns = ""
            self.id = 0
            self.type = 0
            self.action = 0
            self.pose = _PoseStub()
            self.scale = _ScaleStub()
            self.color = ColorRGBA()
            self.points: List[Point] = []
            self.text = ""

    class MarkerArray:  # type: ignore
        def __init__(self) -> None:
            self.markers: List[Marker] = []

    class Vector3:  # type: ignore
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class ConeObservation:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.position = Point()
            self.covariance = [0.0] * 9
            self.class_probs = [0.25, 0.25, 0.25, 0.25]
            self.quality_score = 0.0
            self.sensor_health = 0
            self.source = ""

    class ConeTrack:  # type: ignore
        TRACK_TENTATIVE = 0
        TRACK_CONFIRMED = 1
        TRACK_LOST = 2

        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.track_id = 0
            self.position = Point()
            self.velocity = Vector3()
            self.covariance = [0.0] * 9
            self.class_probs = [0.25, 0.25, 0.25, 0.25]
            self.track_age = 0.0
            self.track_status = 0
            self.quality_score = 0.0

    class CameraConeDetections:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.detections: List[Any] = []
            self.image_quality = 1.0
            self.sensor_health = 0

    class LidarConeCluster:  # type: ignore
        def __init__(self) -> None:
            self.centroid = Point()
            self.covariance = [0.0] * 9
            self.point_count = 0
            self.geometry_quality = 1.0
            self.stripe_class_probs = [0.25, 0.25, 0.25, 0.25]

    class LidarConeClusters:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.clusters: List[LidarConeCluster] = []
            self.sensor_health = 0

    class Odometry:  # type: ignore
        pass

    class TwistStamped:  # type: ignore
        pass

    class TwistWithCovarianceStamped:  # type: ignore
        pass

    class CameraInfo:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.width = 0
            self.height = 0
            self.k = [0.0] * 9
            self.d: List[float] = []
            self.p = [0.0] * 12

    class Image:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.width = 0
            self.height = 0
            self.encoding = ""
            self.step = 0
            self.data = b""

    class DiagnosticArray:  # type: ignore
        def __init__(self) -> None:
            self.header = _HeaderStub()
            self.status = []

    class DiagnosticStatus:  # type: ignore
        OK = 0
        WARN = 1
        ERROR = 2

        def __init__(self) -> None:
            self.name = ""
            self.hardware_id = ""
            self.level = 0
            self.message = ""
            self.values = []

    class KeyValue:  # type: ignore
        def __init__(self, key: str = "", value: str = "") -> None:
            self.key = key
            self.value = value

    class Duration:  # type: ignore
        def __init__(self, seconds: float = 0.0) -> None:
            self.seconds = seconds

    class Time:  # type: ignore
        @staticmethod
        def from_msg(_msg):
            return None

    class QoSProfile:  # type: ignore
        def __init__(self, depth: int = 10) -> None:
            self.depth = depth

    class ReliabilityPolicy:  # type: ignore
        BEST_EFFORT = 0
        RELIABLE = 1

    class HistoryPolicy:  # type: ignore
        KEEP_LAST = 0

    class DurabilityPolicy:  # type: ignore
        VOLATILE = 0

    class Buffer:  # type: ignore
        pass

    class TransformException(Exception):  # type: ignore
        pass

    class TransformListener:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

from .auto_calibrate import (
    CalibrationFrameSample,
    calibrate_lidar_to_camera_extrinsics,
    calibrate_temporal_sync_offset,
)
from .debug_visualizer import (
    build_reprojection_debug_annotations,
    build_scene_marker_descriptors,
)
from .ekf_tracker import (
    CorridorEstimate,
    EgoMotionStep,
    MultiTargetEKFTracker,
    Observation2D,
    TrackEstimate,
    TrackerConfig,
    TrackerSnapshot,
)
from .early_fusion import (
    CameraIntrinsics,
    CameraSensorProfile,
    EarlyFusionProjector,
    LidarToCameraExtrinsics,
    MultiCameraEarlyFusion,
    StereoDepthFrame,
)


SENSOR_HEALTH_OK = 0
SENSOR_HEALTH_DEGRADED = 1
SENSOR_HEALTH_CRITICAL = 2


@dataclass
class LidarSensorProfile:
    lidar_id: str
    topic: str
    frame_id: str
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    offset_z_m: float = 0.0
    yaw_rad: float = 0.0


@dataclass
class FrameRecord:
    stamp: float
    lidar_msg: LidarConeClusters
    snapshot_before: Optional[TrackerSnapshot]
    snapshot_after: Optional[TrackerSnapshot]


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")

        self._declare_parameters()
        self._load_parameters()
        self.projector = self._build_projector()

        sensor_qos = QoSProfile(depth=self.sensor_qos_depth)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.history = HistoryPolicy.KEEP_LAST
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        output_qos = QoSProfile(depth=self.output_qos_depth)
        output_qos.reliability = ReliabilityPolicy.RELIABLE
        output_qos.history = HistoryPolicy.KEEP_LAST
        output_qos.durability = DurabilityPolicy.VOLATILE

        diag_qos = QoSProfile(depth=self.diag_qos_depth)
        diag_qos.reliability = ReliabilityPolicy.RELIABLE
        diag_qos.history = HistoryPolicy.KEEP_LAST
        diag_qos.durability = DurabilityPolicy.VOLATILE

        self.obs_pub = self.create_publisher(ConeObservation, self.obs_out_topic, output_qos)
        self.track_pub = self.create_publisher(ConeTrack, self.track_out_topic, output_qos)
        self.diag_pub = self.create_publisher(DiagnosticArray, self.diag_topic, diag_qos)
        self.corridor_markers_pub = self.create_publisher(
            MarkerArray,
            self.corridor_markers_topic or "/perception/corridor_markers",
            output_qos,
        )
        self.corridor_path_pub = self.create_publisher(
            Path,
            self.corridor_path_topic or "/perception/corridor_path",
            output_qos,
        )
        self.debug_markers_pub = self.create_publisher(
            MarkerArray,
            self.debug_markers_topic or "/perception/debug_markers",
            output_qos,
        )
        self.debug_reprojection_pub = self.create_publisher(
            String,
            self.debug_reprojection_topic or "/perception/debug_reprojection",
            output_qos,
        )

        self.camera_buffer: Deque[Tuple[float, CameraConeDetections]] = deque(
            maxlen=self.queue_size_camera
        )
        self.lidar_buffer: Deque[Tuple[float, LidarConeClusters]] = deque(
            maxlen=self.queue_size_lidar
        )
        self.motion_buffer: Deque[Tuple[float, Any]] = deque(maxlen=self.queue_size_motion)

        self._build_camera_and_lidar_registries()

        for cam_id, cam_profile in self.camera_profiles.items():
            if cam_id == "primary":
                self.create_subscription(
                    CameraConeDetections, cam_profile.topic, self._on_camera, sensor_qos
                )
            else:
                self.create_subscription(
                    CameraConeDetections,
                    cam_profile.topic,
                    lambda msg, cid=cam_id: self._on_camera(msg, camera_id=cid),
                    sensor_qos,
                )
            if (
                getattr(self, "enable_camera_info_intrinsics", True)
                and getattr(cam_profile, "camera_info_topic", "")
            ):
                if cam_id == "primary":
                    self.create_subscription(
                        CameraInfo,
                        cam_profile.camera_info_topic,
                        self._on_camera_info,
                        sensor_qos,
                    )
                else:
                    self.create_subscription(
                        CameraInfo,
                        cam_profile.camera_info_topic,
                        lambda msg, cid=cam_id: self._on_camera_info(msg, camera_id=cid),
                        sensor_qos,
                    )
            if (
                getattr(self, "enable_stereo_depth_fusion", True)
                and getattr(cam_profile, "depth_topic", "")
            ):
                if cam_id == "primary":
                    self.create_subscription(
                        Image,
                        cam_profile.depth_topic,
                        self._on_depth,
                        sensor_qos,
                    )
                else:
                    self.create_subscription(
                        Image,
                        cam_profile.depth_topic,
                        lambda msg, cid=cam_id: self._on_depth(msg, camera_id=cid),
                        sensor_qos,
                    )
        self.create_subscription(
            LidarConeClusters, self.lidar_topic, self._on_lidar, sensor_qos
        )
        for lid_id, lid_profile in self.lidar_profiles.items():
            if lid_id == "primary":
                continue
            self.create_subscription(
                LidarConeClusters,
                lid_profile.topic,
                lambda msg, lid=lid_id: self._on_extra_lidar(msg, lidar_id=lid),
                sensor_qos,
            )
        motion_mtype = str(getattr(self, "motion_msg_type", "odometry")).strip().lower()
        if motion_mtype == "twist_stamped":
            self.create_subscription(TwistStamped, self.motion_topic, self._on_motion, sensor_qos)
        elif motion_mtype == "twist_with_covariance_stamped":
            self.create_subscription(
                TwistWithCovarianceStamped, self.motion_topic, self._on_motion, sensor_qos
            )
        else:
            self.create_subscription(Odometry, self.motion_topic, self._on_motion, sensor_qos)
        self.tracker = MultiTargetEKFTracker(self._tracker_config())
        self.frame_history: List[FrameRecord] = []
        self.history_base_snapshot = self.tracker.empty_snapshot()

        self.last_processed_lidar_stamp = 0.0
        self.last_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self.last_obs_count = 0
        self.last_health = SENSOR_HEALTH_OK
        self.pending_replay = False
        self.replay_anchor_stamp: Optional[float] = None
        self.last_ego_speed_mps = 0.0
        self.last_blind_distance_current_m = 0.0
        self.last_blind_distance_target_m = 0.0
        self.max_blind_distance_target_m = 0.0
        self.last_camera_lidar_delta_ms = 0.0
        self.last_camera_lidar_abs_delta_ms = 0.0
        self.max_camera_lidar_abs_delta_ms = 0.0
        self.camera_lidar_delta_samples = 0
        self.sync_delta_mean_ms = 0.0
        self.sync_delta_std_ms = 0.0
        self.sync_delta_abs_p95_ms = 0.0
        self.last_projection_residual_mean_px = 0.0
        self.projection_residual_sum_px_total = 0.0
        self.projection_residual_samples_total = 0
        self.projection_residual_max_px = 0.0
        self.last_motion_age_ms = 0.0
        self.max_motion_age_ms = 0.0
        self.last_motion_comp_error_m = 0.0
        self.max_motion_comp_error_m = 0.0
        self.last_camera_fallback_stamp = -1.0
        self.sync_delta_window_ms: Deque[float] = deque(
            maxlen=max(4, self.sync_stats_window_size)
        )
        self.last_tf_lookup_ok = False
        self.last_tf_lookup_error = ""
        self.last_tf_age_ms = 0.0
        self.last_extrinsics_source = "params"
        self.last_extrinsics_sanity = "ok"
        self.last_tf_lidar_frame = ""
        self.last_tf_camera_frame = ""
        self.last_stage_ms = {
            "camera_lookup": 0.0,
            "extrinsics_lookup": 0.0,
            "build_observations": 0.0,
            "publish_observations": 0.0,
            "tracker_step": 0.0,
            "replay_total": 0.0,
            "cycle_total": 0.0,
        }
        self.max_stage_ms = {key: 0.0 for key in self.last_stage_ms}

        self.counters = {
            "camera_overflow": 0,
            "lidar_overflow": 0,
            "motion_overflow": 0,
            "oosm_within_budget": 0,
            "oosm_dropped": 0,
            "camera_miss": 0,
            "tracks_created": 0,
            "tracks_deleted": 0,
            "assoc_candidates": 0,
            "assoc_matches": 0,
            "assoc_rejects": 0,
            "replay_requests": 0,
            "replay_completed": 0,
            "replay_frames": 0,
            "replay_budget_exceeded": 0,
            "replay_no_frames": 0,
            "replay_no_base_snapshot": 0,
            "projection_projected_samples": 0,
            "projection_matched_samples": 0,
            "projection_clusters_no_match": 0,
            "projection_high_residual_clusters": 0,
            "frustum_recovered_observations": 0,
            "stereo_depth_frames": 0,
            "stereo_depth_fused_observations": 0,
            "camera_fallback_frames": 0,
            "camera_fallback_observations": 0,
            "camera_fallback_no_camera": 0,
            "camera_fallback_stale_camera": 0,
            "camera_fallback_duplicate_frame": 0,
            "tf_lookup_success": 0,
            "tf_lookup_fail": 0,
            "tf_extrinsics_stale": 0,
            "tf_extrinsics_missing_frame": 0,
            "motion_missing_for_covariance": 0,
            "online_calib_runs": 0,
            "online_calib_applied": 0,
            "corridor_published": 0,
            "debug_markers_published": 0,
            "extra_lidar_messages": 0,
            "camera_info_updates": 0,
        }
        self.last_corridor_estimate: Optional[CorridorEstimate] = None
        self.last_debug_markers: List[Any] = []
        self.last_reprojection_annotations: List[Any] = []
        self.last_camera_info_cameras: Dict[str, Tuple[float, float, float, float, int, int]] = {}
        self.calibration_window: Deque[CalibrationFrameSample] = deque(
            maxlen=max(5, int(getattr(self, "online_calibration_window_frames", 30)))
        )
        self._online_calib_frames_since_run = 0
        self.last_online_calibration_report: Optional[Dict[str, Any]] = None

        self.tf_buffer: Optional[Buffer] = None
        self.tf_listener: Optional[TransformListener] = None
        if self.use_tf_extrinsics:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.create_timer(1.0 / max(1.0, self.process_rate_hz), self._process_once)
        self.create_timer(1.0 / max(0.5, self.diag_rate_hz), self._publish_diagnostics)

        self.get_logger().info("fusion_node started")
        self.get_logger().info(
            "Validation note: verify calibrated extrinsics and clock sync on target vehicle before track deployment."
        )
        self.get_logger().info(
            "projection semantics: "
            f"{'enabled' if self.enable_projection_semantics else 'disabled'} "
            f"(fx={self.camera_fx:.1f}, fy={self.camera_fy:.1f}, "
            f"size={self.camera_width}x{self.camera_height})"
        )
        self.get_logger().info(
            "extrinsics source: "
            f"{'tf2 runtime lookup' if self.use_tf_extrinsics else 'static parameters'}"
        )

    def _declare_parameters(self) -> None:
        defaults = [
            ("camera_topic", "/camera/cone_detections"),
            ("lidar_topic", "/lidar/cone_clusters"),
            ("motion_topic", "/ego/odometry"),
            ("motion_msg_type", "odometry"),
            ("obs_out_topic", "/perception/cone_observations"),
            ("track_out_topic", "/perception/cone_tracks"),
            ("diag_topic", "/perception/diagnostics"),
            ("target_frame", "base_link"),
            ("enable_projection_semantics", True),
            ("camera_fx", 700.0),
            ("camera_fy", 700.0),
            ("camera_cx", 640.0),
            ("camera_cy", 360.0),
            ("camera_width", 1280),
            ("camera_height", 720),
            ("camera_distortion_model", "plumb_bob"),
            ("camera_k1", 0.0),
            ("camera_k2", 0.0),
            ("camera_p1", 0.0),
            ("camera_p2", 0.0),
            ("camera_k3", 0.0),
            ("camera_k4", 0.0),
            ("lidar_to_camera_tx_m", 0.0),
            ("lidar_to_camera_ty_m", 0.0),
            ("lidar_to_camera_tz_m", 0.0),
            ("lidar_to_camera_roll_rad", -3.141592653589793),
            ("lidar_to_camera_pitch_rad", -1.5707963267948966),
            ("lidar_to_camera_yaw_rad", -1.5707963267948966),
            ("use_tf_extrinsics", False),
            ("tf_extrinsics_lidar_frame", ""),
            ("tf_extrinsics_camera_frame", ""),
            ("tf_lookup_timeout_ms", 20.0),
            ("tf_extrinsics_warn_age_ms", 40.0),
            ("tf_extrinsics_critical_age_ms", 120.0),
            ("projection_sample_radius_scale", 1.5),
            ("projection_min_depth_m", 0.1),
            ("projection_max_depth_m", 80.0),
            ("projection_min_semantic_weight", 0.35),
            ("projection_no_match_semantic_weight", 0.6),
            ("projection_soft_likelihood_enabled", True),
            ("projection_soft_likelihood_sigma_scale", 0.35),
            ("projection_soft_likelihood_min_value", 0.02),
            ("projection_soft_likelihood_outside_ratio", 1.6),
            ("projection_residual_warn_px", 5.0),
            ("projection_residual_critical_px", 12.0),
            ("max_rewind_ms", 150),
            ("max_replay_steps", 8),
            ("frame_history_size", 128),
            ("max_camera_lidar_delta_ms", 120),
            ("sync_offset_warn_ms", 40.0),
            ("sync_offset_critical_ms", 90.0),
            ("sync_stats_window_size", 60),
            ("sync_jitter_warn_ms", 10.0),
            ("sync_jitter_critical_ms", 25.0),
            ("camera_retention_s", 2.0),
            ("enable_camera_fallback", True),
            ("camera_fallback_max_age_s", 0.25),
            ("camera_fallback_min_confidence", 0.25),
            ("camera_fallback_min_bbox_height_px", 8.0),
            ("camera_fallback_cone_height_m", 0.30),
            ("camera_fallback_min_depth_m", 2.0),
            ("camera_fallback_max_depth_m", 35.0),
            ("camera_fallback_quality_scale", 0.45),
            ("camera_fallback_cov_xy_m2", 4.0),
            ("camera_fallback_cov_z_m2", 8.0),
            ("queue_size_camera", 16),
            ("queue_size_lidar", 16),
            ("queue_size_motion", 64),
            ("drop_policy", "drop_oldest"),
            ("blind_distance_target_speed_mps", 16.0),
            ("motion_age_warn_ms", 80.0),
            ("motion_age_critical_ms", 180.0),
            ("motion_comp_error_gain", 0.6),
            ("process_rate_hz", 30.0),
            ("diag_rate_hz", 2.0),
            ("enable_tracks", True),
            ("track_mahalanobis_gate", 9.21),
            ("track_semantic_min_overlap", 0.15),
            ("track_semantic_cost_weight", 2.0),
            ("track_process_noise_accel", 2.0),
            ("track_min_meas_var", 0.05),
            ("track_confirm_hits", 3),
            ("track_max_missed", 5),
            ("track_class_update_alpha", 0.35),
            ("track_max_predict_dt_s", 0.25),
            ("track_motion_model", "landmark"),
            ("track_assignment_algorithm", "hungarian"),
            ("track_semantic_model", "dirichlet"),
            ("track_dirichlet_prior_strength", 1.0),
            ("track_dirichlet_evidence_scale", 4.0),
            ("track_dirichlet_max_concentration", 40.0),
            ("track_landmark_process_noise_pos", 0.03),
            ("track_odom_vx_std", 0.15),
            ("track_odom_vy_std", 0.15),
            ("track_odom_wz_std", 0.05),
            ("track_corridor_prior_weight", 0.0),
            ("enable_global_map", False),
            ("global_map_gate_radius_m", 1.5),
            ("global_map_min_lap_distance_m", 40.0),
            ("enable_lidar_stripe_semantics", True),
            ("enable_corridor_output", False),
            ("corridor_markers_topic", "/perception/corridor_markers"),
            ("corridor_path_topic", "/perception/corridor_path"),
            ("corridor_horizon_m", 20.0),
            ("corridor_step_m", 1.0),
            ("enable_debug_viz", False),
            ("debug_markers_topic", "/perception/debug_markers"),
            ("debug_reprojection_topic", "/perception/debug_reprojection"),
            ("enable_online_calibration", False),
            ("online_calibration_window_frames", 30),
            ("online_calibration_min_pairs", 12),
            ("online_calibration_apply_extrinsics", False),
            ("secondary_camera_topic", ""),
            ("enable_camera_info_intrinsics", True),
            ("camera_info_topic", "/camera/camera_info"),
            ("secondary_camera_info_topic", ""),
            ("camera_profiles_json", ""),
            ("secondary_camera_fx", 0.0),
            ("secondary_camera_fy", 0.0),
            ("secondary_camera_cx", 0.0),
            ("secondary_camera_cy", 0.0),
            ("secondary_lidar_to_camera_tx_m", 0.0),
            ("secondary_lidar_to_camera_ty_m", 0.0),
            ("secondary_lidar_to_camera_tz_m", 0.0),
            ("secondary_lidar_to_camera_roll_rad", 0.0),
            ("secondary_lidar_to_camera_pitch_rad", 0.0),
            ("secondary_lidar_to_camera_yaw_rad", 0.0),
            ("lidar_topics_extra", [""]),
            ("lidar_profiles_json", ""),
            ("require_camera_for_single_point_clusters", True),
            ("assume_uncounted_clusters_valid", True),
            ("min_uncorroborated_cluster_points", 2),
            ("enable_chassis_attitude_compensation", True),
            ("enable_bidirectional_frustum_fusion", True),
            ("enable_hungarian_early_fusion", True),
            ("enable_frustum_hull_sampling", False),
            ("enable_triangular_cone_mask", True),
            ("cone_mask_apex_width_ratio", 0.18),
            ("enable_ground_plane_depth_fusion", True),
            ("ground_z_base_m", -0.22),
            ("ground_slope_rad", 0.0),
            ("enable_stereo_depth_fusion", True),
            ("depth_topic", ""),
            ("secondary_depth_topic", ""),
            ("stereo_depth_scale", 0.0),
            ("stereo_baseline_m", 0.12),
            ("stereo_disparity_sigma_px", 0.5),
            ("stereo_depth_max_age_ms", 150.0),
            ("stereo_depth_min_m", 0.5),
            ("stereo_depth_max_m", 45.0),
            ("frustum_min_confidence", 0.55),
            ("frustum_dedup_radius_m", 1.2),
            ("aero_pitch_rad_per_mps2", 0.0),
            ("aero_tz_m_per_mps2", 0.0),
            ("health_degrade_scale", 1.5),
            ("quality_warn_threshold", 0.4),
            ("quality_critical_threshold", 0.2),
            ("sensor_qos_depth", 10),
            ("output_qos_depth", 20),
            ("diag_qos_depth", 10),
        ]
        for name, value in defaults:
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.motion_topic = str(self.get_parameter("motion_topic").value)
        self.motion_msg_type = str(self.get_parameter("motion_msg_type").value)
        self.obs_out_topic = str(self.get_parameter("obs_out_topic").value)
        self.track_out_topic = str(self.get_parameter("track_out_topic").value)
        self.diag_topic = str(self.get_parameter("diag_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.enable_projection_semantics = bool(
            self.get_parameter("enable_projection_semantics").value
        )
        self.camera_fx = float(self.get_parameter("camera_fx").value)
        self.camera_fy = float(self.get_parameter("camera_fy").value)
        self.camera_cx = float(self.get_parameter("camera_cx").value)
        self.camera_cy = float(self.get_parameter("camera_cy").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_distortion_model = str(
            self.get_parameter("camera_distortion_model").value
        )
        self.camera_k1 = float(self.get_parameter("camera_k1").value)
        self.camera_k2 = float(self.get_parameter("camera_k2").value)
        self.camera_p1 = float(self.get_parameter("camera_p1").value)
        self.camera_p2 = float(self.get_parameter("camera_p2").value)
        self.camera_k3 = float(self.get_parameter("camera_k3").value)
        self.camera_k4 = float(self.get_parameter("camera_k4").value)
        self.lidar_to_camera_tx_m = float(
            self.get_parameter("lidar_to_camera_tx_m").value
        )
        self.lidar_to_camera_ty_m = float(
            self.get_parameter("lidar_to_camera_ty_m").value
        )
        self.lidar_to_camera_tz_m = float(
            self.get_parameter("lidar_to_camera_tz_m").value
        )
        self.lidar_to_camera_roll_rad = float(
            self.get_parameter("lidar_to_camera_roll_rad").value
        )
        self.lidar_to_camera_pitch_rad = float(
            self.get_parameter("lidar_to_camera_pitch_rad").value
        )
        self.lidar_to_camera_yaw_rad = float(
            self.get_parameter("lidar_to_camera_yaw_rad").value
        )
        self.use_tf_extrinsics = bool(self.get_parameter("use_tf_extrinsics").value)
        self.tf_extrinsics_lidar_frame = str(
            self.get_parameter("tf_extrinsics_lidar_frame").value
        )
        self.tf_extrinsics_camera_frame = str(
            self.get_parameter("tf_extrinsics_camera_frame").value
        )
        self.tf_lookup_timeout_ms = float(self.get_parameter("tf_lookup_timeout_ms").value)
        self.tf_extrinsics_warn_age_ms = float(
            self.get_parameter("tf_extrinsics_warn_age_ms").value
        )
        self.tf_extrinsics_critical_age_ms = float(
            self.get_parameter("tf_extrinsics_critical_age_ms").value
        )
        self.projection_sample_radius_scale = float(
            self.get_parameter("projection_sample_radius_scale").value
        )
        self.projection_min_depth_m = float(
            self.get_parameter("projection_min_depth_m").value
        )
        self.projection_max_depth_m = float(
            self.get_parameter("projection_max_depth_m").value
        )
        self.projection_min_semantic_weight = float(
            self.get_parameter("projection_min_semantic_weight").value
        )
        self.projection_no_match_semantic_weight = float(
            self.get_parameter("projection_no_match_semantic_weight").value
        )
        self.projection_soft_likelihood_enabled = bool(
            self.get_parameter("projection_soft_likelihood_enabled").value
        )
        self.projection_soft_likelihood_sigma_scale = float(
            self.get_parameter("projection_soft_likelihood_sigma_scale").value
        )
        self.projection_soft_likelihood_min_value = float(
            self.get_parameter("projection_soft_likelihood_min_value").value
        )
        self.projection_soft_likelihood_outside_ratio = float(
            self.get_parameter("projection_soft_likelihood_outside_ratio").value
        )
        self.projection_residual_warn_px = float(
            self.get_parameter("projection_residual_warn_px").value
        )
        self.projection_residual_critical_px = float(
            self.get_parameter("projection_residual_critical_px").value
        )
        self.max_rewind_s = float(self.get_parameter("max_rewind_ms").value) / 1000.0
        self.max_replay_steps = int(self.get_parameter("max_replay_steps").value)
        self.frame_history_size = int(self.get_parameter("frame_history_size").value)
        self.max_camera_lidar_delta_s = (
            float(self.get_parameter("max_camera_lidar_delta_ms").value) / 1000.0
        )
        self.sync_offset_warn_ms = float(self.get_parameter("sync_offset_warn_ms").value)
        self.sync_offset_critical_ms = float(
            self.get_parameter("sync_offset_critical_ms").value
        )
        self.sync_stats_window_size = int(
            self.get_parameter("sync_stats_window_size").value
        )
        self.sync_jitter_warn_ms = float(
            self.get_parameter("sync_jitter_warn_ms").value
        )
        self.sync_jitter_critical_ms = float(
            self.get_parameter("sync_jitter_critical_ms").value
        )
        self.camera_retention_s = float(self.get_parameter("camera_retention_s").value)
        self.enable_camera_fallback = bool(
            self.get_parameter("enable_camera_fallback").value
        )
        self.camera_fallback_max_age_s = float(
            self.get_parameter("camera_fallback_max_age_s").value
        )
        self.camera_fallback_min_confidence = float(
            self.get_parameter("camera_fallback_min_confidence").value
        )
        self.camera_fallback_min_bbox_height_px = float(
            self.get_parameter("camera_fallback_min_bbox_height_px").value
        )
        self.camera_fallback_cone_height_m = float(
            self.get_parameter("camera_fallback_cone_height_m").value
        )
        self.camera_fallback_min_depth_m = float(
            self.get_parameter("camera_fallback_min_depth_m").value
        )
        self.camera_fallback_max_depth_m = float(
            self.get_parameter("camera_fallback_max_depth_m").value
        )
        self.camera_fallback_quality_scale = float(
            self.get_parameter("camera_fallback_quality_scale").value
        )
        self.camera_fallback_cov_xy_m2 = float(
            self.get_parameter("camera_fallback_cov_xy_m2").value
        )
        self.camera_fallback_cov_z_m2 = float(
            self.get_parameter("camera_fallback_cov_z_m2").value
        )
        self.queue_size_camera = int(self.get_parameter("queue_size_camera").value)
        self.queue_size_lidar = int(self.get_parameter("queue_size_lidar").value)
        self.queue_size_motion = int(self.get_parameter("queue_size_motion").value)
        self.drop_policy = str(self.get_parameter("drop_policy").value)
        self.blind_distance_target_speed_mps = float(
            self.get_parameter("blind_distance_target_speed_mps").value
        )
        self.motion_age_warn_ms = float(self.get_parameter("motion_age_warn_ms").value)
        self.motion_age_critical_ms = float(
            self.get_parameter("motion_age_critical_ms").value
        )
        self.motion_comp_error_gain = float(
            self.get_parameter("motion_comp_error_gain").value
        )
        self.process_rate_hz = float(self.get_parameter("process_rate_hz").value)
        self.diag_rate_hz = float(self.get_parameter("diag_rate_hz").value)
        self.enable_tracks = bool(self.get_parameter("enable_tracks").value)
        self.track_mahalanobis_gate = float(self.get_parameter("track_mahalanobis_gate").value)
        self.track_semantic_min_overlap = float(
            self.get_parameter("track_semantic_min_overlap").value
        )
        self.track_semantic_cost_weight = float(
            self.get_parameter("track_semantic_cost_weight").value
        )
        self.track_process_noise_accel = float(
            self.get_parameter("track_process_noise_accel").value
        )
        self.track_min_meas_var = float(self.get_parameter("track_min_meas_var").value)
        self.track_confirm_hits = int(self.get_parameter("track_confirm_hits").value)
        self.track_max_missed = int(self.get_parameter("track_max_missed").value)
        self.track_class_update_alpha = float(
            self.get_parameter("track_class_update_alpha").value
        )
        self.track_max_predict_dt_s = float(
            self.get_parameter("track_max_predict_dt_s").value
        )
        self.track_motion_model = str(self.get_parameter("track_motion_model").value)
        self.track_assignment_algorithm = str(
            self.get_parameter("track_assignment_algorithm").value
        )
        self.track_semantic_model = str(self.get_parameter("track_semantic_model").value)
        self.track_dirichlet_prior_strength = float(
            self.get_parameter("track_dirichlet_prior_strength").value
        )
        self.track_dirichlet_evidence_scale = float(
            self.get_parameter("track_dirichlet_evidence_scale").value
        )
        self.track_dirichlet_max_concentration = float(
            self.get_parameter("track_dirichlet_max_concentration").value
        )
        self.track_landmark_process_noise_pos = float(
            self.get_parameter("track_landmark_process_noise_pos").value
        )
        self.track_odom_vx_std = float(self.get_parameter("track_odom_vx_std").value)
        self.track_odom_vy_std = float(self.get_parameter("track_odom_vy_std").value)
        self.track_odom_wz_std = float(self.get_parameter("track_odom_wz_std").value)
        self.track_corridor_prior_weight = float(
            self.get_parameter("track_corridor_prior_weight").value
        )
        self.enable_global_map = bool(self.get_parameter("enable_global_map").value)
        self.global_map_gate_radius_m = float(
            self.get_parameter("global_map_gate_radius_m").value
        )
        self.global_map_min_lap_distance_m = float(
            self.get_parameter("global_map_min_lap_distance_m").value
        )
        self.enable_lidar_stripe_semantics = bool(
            self.get_parameter("enable_lidar_stripe_semantics").value
        )
        self.enable_corridor_output = bool(
            self.get_parameter("enable_corridor_output").value
        )
        self.corridor_markers_topic = str(
            self.get_parameter("corridor_markers_topic").value
        ).strip()
        self.corridor_path_topic = str(
            self.get_parameter("corridor_path_topic").value
        ).strip()
        self.corridor_horizon_m = float(
            self.get_parameter("corridor_horizon_m").value
        )
        self.corridor_step_m = float(
            self.get_parameter("corridor_step_m").value
        )
        self.enable_debug_viz = bool(
            self.get_parameter("enable_debug_viz").value
        )
        self.debug_markers_topic = str(
            self.get_parameter("debug_markers_topic").value
        ).strip()
        self.debug_reprojection_topic = str(
            self.get_parameter("debug_reprojection_topic").value
        ).strip()
        self.enable_online_calibration = bool(
            self.get_parameter("enable_online_calibration").value
        )
        self.online_calibration_window_frames = max(
            5, int(self.get_parameter("online_calibration_window_frames").value)
        )
        self.online_calibration_min_pairs = max(
            3, int(self.get_parameter("online_calibration_min_pairs").value)
        )
        self.online_calibration_apply_extrinsics = bool(
            self.get_parameter("online_calibration_apply_extrinsics").value
        )
        self.secondary_camera_topic = str(
            self.get_parameter("secondary_camera_topic").value
        ).strip()
        self.enable_camera_info_intrinsics = bool(
            self.get_parameter("enable_camera_info_intrinsics").value
        )
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        ).strip()
        self.secondary_camera_info_topic = str(
            self.get_parameter("secondary_camera_info_topic").value
        ).strip()
        self.camera_profiles_json = str(
            self.get_parameter("camera_profiles_json").value
        ).strip()
        self.secondary_camera_fx = float(
            self.get_parameter("secondary_camera_fx").value
        )
        self.secondary_camera_fy = float(
            self.get_parameter("secondary_camera_fy").value
        )
        self.secondary_camera_cx = float(
            self.get_parameter("secondary_camera_cx").value
        )
        self.secondary_camera_cy = float(
            self.get_parameter("secondary_camera_cy").value
        )
        self.secondary_lidar_to_camera_tx_m = float(
            self.get_parameter("secondary_lidar_to_camera_tx_m").value
        )
        self.secondary_lidar_to_camera_ty_m = float(
            self.get_parameter("secondary_lidar_to_camera_ty_m").value
        )
        self.secondary_lidar_to_camera_tz_m = float(
            self.get_parameter("secondary_lidar_to_camera_tz_m").value
        )
        self.secondary_lidar_to_camera_roll_rad = float(
            self.get_parameter("secondary_lidar_to_camera_roll_rad").value
        )
        self.secondary_lidar_to_camera_pitch_rad = float(
            self.get_parameter("secondary_lidar_to_camera_pitch_rad").value
        )
        self.secondary_lidar_to_camera_yaw_rad = float(
            self.get_parameter("secondary_lidar_to_camera_yaw_rad").value
        )
        raw_lidar_extra = self.get_parameter("lidar_topics_extra").value
        if isinstance(raw_lidar_extra, str):
            self.lidar_topics_extra = [
                part.strip() for part in raw_lidar_extra.split(",") if part.strip()
            ]
        elif isinstance(raw_lidar_extra, (list, tuple)):
            self.lidar_topics_extra = [
                str(part).strip() for part in raw_lidar_extra if str(part).strip()
            ]
        else:
            self.lidar_topics_extra = []
        self.lidar_profiles_json = str(
            self.get_parameter("lidar_profiles_json").value
        ).strip()
        self.require_camera_for_single_point_clusters = bool(
            self.get_parameter("require_camera_for_single_point_clusters").value
        )
        self.assume_uncounted_clusters_valid = bool(
            self.get_parameter("assume_uncounted_clusters_valid").value
        )
        self.min_uncorroborated_cluster_points = int(
            self.get_parameter("min_uncorroborated_cluster_points").value
        )
        self.enable_chassis_attitude_compensation = bool(
            self.get_parameter("enable_chassis_attitude_compensation").value
        )
        self.enable_bidirectional_frustum_fusion = bool(
            self.get_parameter("enable_bidirectional_frustum_fusion").value
        )
        self.enable_hungarian_early_fusion = bool(
            self.get_parameter("enable_hungarian_early_fusion").value
        )
        self.enable_frustum_hull_sampling = bool(
            self.get_parameter("enable_frustum_hull_sampling").value
        )
        self.enable_triangular_cone_mask = bool(
            self.get_parameter("enable_triangular_cone_mask").value
        )
        self.cone_mask_apex_width_ratio = float(
            self.get_parameter("cone_mask_apex_width_ratio").value
        )
        self.enable_ground_plane_depth_fusion = bool(
            self.get_parameter("enable_ground_plane_depth_fusion").value
        )
        self.ground_z_base_m = float(
            self.get_parameter("ground_z_base_m").value
        )
        self.ground_slope_rad = float(
            self.get_parameter("ground_slope_rad").value
        )
        self.enable_stereo_depth_fusion = bool(
            self.get_parameter("enable_stereo_depth_fusion").value
        )
        self.depth_topic = str(self.get_parameter("depth_topic").value).strip()
        self.secondary_depth_topic = str(
            self.get_parameter("secondary_depth_topic").value
        ).strip()
        self.stereo_depth_scale = float(
            self.get_parameter("stereo_depth_scale").value
        )
        self.stereo_baseline_m = float(
            self.get_parameter("stereo_baseline_m").value
        )
        self.stereo_disparity_sigma_px = float(
            self.get_parameter("stereo_disparity_sigma_px").value
        )
        self.stereo_depth_max_age_ms = float(
            self.get_parameter("stereo_depth_max_age_ms").value
        )
        self.stereo_depth_min_m = float(
            self.get_parameter("stereo_depth_min_m").value
        )
        self.stereo_depth_max_m = float(
            self.get_parameter("stereo_depth_max_m").value
        )
        self.frustum_min_confidence = float(
            self.get_parameter("frustum_min_confidence").value
        )
        self.frustum_dedup_radius_m = float(
            self.get_parameter("frustum_dedup_radius_m").value
        )
        self.aero_pitch_rad_per_mps2 = float(
            self.get_parameter("aero_pitch_rad_per_mps2").value
        )
        self.aero_tz_m_per_mps2 = float(self.get_parameter("aero_tz_m_per_mps2").value)
        self.health_degrade_scale = float(self.get_parameter("health_degrade_scale").value)
        self.quality_warn_threshold = float(
            self.get_parameter("quality_warn_threshold").value
        )
        self.quality_critical_threshold = float(
            self.get_parameter("quality_critical_threshold").value
        )
        self.sensor_qos_depth = int(self.get_parameter("sensor_qos_depth").value)
        self.output_qos_depth = int(self.get_parameter("output_qos_depth").value)
        self.diag_qos_depth = int(self.get_parameter("diag_qos_depth").value)

    def _tracker_config(self) -> TrackerConfig:
        return TrackerConfig(
            mahalanobis_gate=self.track_mahalanobis_gate,
            semantic_min_overlap=self.track_semantic_min_overlap,
            semantic_cost_weight=self.track_semantic_cost_weight,
            process_noise_accel=self.track_process_noise_accel,
            min_meas_var=self.track_min_meas_var,
            confirm_hits=self.track_confirm_hits,
            max_missed=self.track_max_missed,
            class_update_alpha=self.track_class_update_alpha,
            max_predict_dt=self.track_max_predict_dt_s,
            motion_model=self.track_motion_model,
            assignment_algorithm=self.track_assignment_algorithm,
            semantic_model=self.track_semantic_model,
            dirichlet_prior_strength=self.track_dirichlet_prior_strength,
            dirichlet_evidence_scale=self.track_dirichlet_evidence_scale,
            dirichlet_max_concentration=self.track_dirichlet_max_concentration,
            landmark_process_noise_pos=self.track_landmark_process_noise_pos,
            odom_vx_std=self.track_odom_vx_std,
            odom_vy_std=self.track_odom_vy_std,
            odom_wz_std=self.track_odom_wz_std,
            corridor_prior_weight=getattr(self, "track_corridor_prior_weight", 0.0),
            enable_global_map=bool(getattr(self, "enable_global_map", False)),
            global_map_gate_radius_m=float(
                getattr(self, "global_map_gate_radius_m", 1.5)
            ),
            global_map_min_lap_distance_m=float(
                getattr(self, "global_map_min_lap_distance_m", 40.0)
            ),
        )

    def _build_projector(self) -> EarlyFusionProjector:
        intrinsics = CameraIntrinsics(
            fx=self.camera_fx,
            fy=self.camera_fy,
            cx=self.camera_cx,
            cy=self.camera_cy,
            width=max(1, self.camera_width),
            height=max(1, self.camera_height),
            k1=getattr(self, "camera_k1", 0.0),
            k2=getattr(self, "camera_k2", 0.0),
            p1=getattr(self, "camera_p1", 0.0),
            p2=getattr(self, "camera_p2", 0.0),
            k3=getattr(self, "camera_k3", 0.0),
            k4=getattr(self, "camera_k4", 0.0),
            distortion_model=str(getattr(self, "camera_distortion_model", "plumb_bob")),
        )
        extrinsics = LidarToCameraExtrinsics.from_rpy_translation(
            tx=self.lidar_to_camera_tx_m,
            ty=self.lidar_to_camera_ty_m,
            tz=self.lidar_to_camera_tz_m,
            roll=self.lidar_to_camera_roll_rad,
            pitch=self.lidar_to_camera_pitch_rad,
            yaw=self.lidar_to_camera_yaw_rad,
        )
        return EarlyFusionProjector(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            min_depth_m=self.projection_min_depth_m,
            max_depth_m=self.projection_max_depth_m,
            sample_radius_scale=self.projection_sample_radius_scale,
            min_semantic_weight=self.projection_min_semantic_weight,
            no_match_semantic_weight=self.projection_no_match_semantic_weight,
            soft_likelihood_enabled=self.projection_soft_likelihood_enabled,
            soft_likelihood_sigma_scale=self.projection_soft_likelihood_sigma_scale,
            soft_likelihood_min_value=self.projection_soft_likelihood_min_value,
            soft_likelihood_outside_ratio=self.projection_soft_likelihood_outside_ratio,
            aero_pitch_rad_per_mps2=self.aero_pitch_rad_per_mps2,
            aero_tz_m_per_mps2=self.aero_tz_m_per_mps2,
            enable_frustum_hull_sampling=getattr(self, "enable_frustum_hull_sampling", False),
            enable_triangular_cone_mask=getattr(self, "enable_triangular_cone_mask", True),
            cone_mask_apex_width_ratio=getattr(self, "cone_mask_apex_width_ratio", 0.18),
        )

    def _make_projector(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: LidarToCameraExtrinsics,
    ) -> EarlyFusionProjector:
        return EarlyFusionProjector(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            min_depth_m=self.projection_min_depth_m,
            max_depth_m=self.projection_max_depth_m,
            sample_radius_scale=self.projection_sample_radius_scale,
            min_semantic_weight=self.projection_min_semantic_weight,
            no_match_semantic_weight=self.projection_no_match_semantic_weight,
            soft_likelihood_enabled=self.projection_soft_likelihood_enabled,
            soft_likelihood_sigma_scale=self.projection_soft_likelihood_sigma_scale,
            soft_likelihood_min_value=self.projection_soft_likelihood_min_value,
            soft_likelihood_outside_ratio=self.projection_soft_likelihood_outside_ratio,
            aero_pitch_rad_per_mps2=self.aero_pitch_rad_per_mps2,
            aero_tz_m_per_mps2=self.aero_tz_m_per_mps2,
            enable_frustum_hull_sampling=getattr(self, "enable_frustum_hull_sampling", False),
            enable_triangular_cone_mask=getattr(self, "enable_triangular_cone_mask", True),
            cone_mask_apex_width_ratio=getattr(self, "cone_mask_apex_width_ratio", 0.18),
        )

    def _build_camera_and_lidar_registries(self) -> None:
        self.camera_profiles: Dict[str, CameraSensorProfile] = {
            "primary": CameraSensorProfile(
                camera_id="primary",
                topic=self.camera_topic,
                frame_id=self.tf_extrinsics_camera_frame or "camera_optical",
                projector=self.projector,
                use_tf_extrinsics=self.use_tf_extrinsics,
                camera_info_topic=getattr(self, "camera_info_topic", ""),
                depth_topic=getattr(self, "depth_topic", ""),
            )
        }
        sec_topic = getattr(self, "secondary_camera_topic", "").strip()
        if sec_topic and sec_topic != self.camera_topic:
            sec_fx = self.secondary_camera_fx if self.secondary_camera_fx > 0.0 else self.camera_fx
            sec_fy = self.secondary_camera_fy if self.secondary_camera_fy > 0.0 else self.camera_fy
            sec_cx = self.secondary_camera_cx if self.secondary_camera_cx > 0.0 else self.camera_cx
            sec_cy = self.secondary_camera_cy if self.secondary_camera_cy > 0.0 else self.camera_cy
            sec_intr = CameraIntrinsics(
                fx=sec_fx,
                fy=sec_fy,
                cx=sec_cx,
                cy=sec_cy,
                width=max(1, self.camera_width),
                height=max(1, self.camera_height),
                k1=getattr(self, "camera_k1", 0.0),
                k2=getattr(self, "camera_k2", 0.0),
                p1=getattr(self, "camera_p1", 0.0),
                p2=getattr(self, "camera_p2", 0.0),
                k3=getattr(self, "camera_k3", 0.0),
                k4=getattr(self, "camera_k4", 0.0),
                distortion_model=str(getattr(self, "camera_distortion_model", "plumb_bob")),
            )
            sec_ext = LidarToCameraExtrinsics.from_rpy_translation(
                tx=getattr(self, "secondary_lidar_to_camera_tx_m", 0.0),
                ty=getattr(self, "secondary_lidar_to_camera_ty_m", 0.0),
                tz=getattr(self, "secondary_lidar_to_camera_tz_m", 0.0),
                roll=getattr(self, "secondary_lidar_to_camera_roll_rad", 0.0),
                pitch=getattr(self, "secondary_lidar_to_camera_pitch_rad", 0.0),
                yaw=getattr(self, "secondary_lidar_to_camera_yaw_rad", 0.0),
            )
            self.camera_profiles["secondary"] = CameraSensorProfile(
                camera_id="secondary",
                topic=sec_topic,
                frame_id="secondary_camera_optical",
                projector=self._make_projector(sec_intr, sec_ext),
                use_tf_extrinsics=False,
                camera_info_topic=getattr(self, "secondary_camera_info_topic", ""),
                depth_topic=getattr(self, "secondary_depth_topic", ""),
            )

        cam_json = getattr(self, "camera_profiles_json", "").strip()
        if cam_json:
            try:
                parsed = json.loads(cam_json)
                if isinstance(parsed, dict):
                    entries = [
                        {"camera_id": k, **v} if isinstance(v, dict) else {"camera_id": k}
                        for k, v in parsed.items()
                    ]
                elif isinstance(parsed, list):
                    entries = parsed
                else:
                    entries = []
                for idx, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    cid = str(entry.get("camera_id", f"cam_{idx}")).strip() or f"cam_{idx}"
                    topic = str(entry.get("topic", "")).strip()
                    if not topic:
                        continue
                    intr = CameraIntrinsics(
                        fx=float(entry.get("fx", self.camera_fx)),
                        fy=float(entry.get("fy", self.camera_fy)),
                        cx=float(entry.get("cx", self.camera_cx)),
                        cy=float(entry.get("cy", self.camera_cy)),
                        width=max(1, int(entry.get("width", self.camera_width))),
                        height=max(1, int(entry.get("height", self.camera_height))),
                        k1=float(entry.get("k1", getattr(self, "camera_k1", 0.0))),
                        k2=float(entry.get("k2", getattr(self, "camera_k2", 0.0))),
                        p1=float(entry.get("p1", getattr(self, "camera_p1", 0.0))),
                        p2=float(entry.get("p2", getattr(self, "camera_p2", 0.0))),
                        k3=float(entry.get("k3", getattr(self, "camera_k3", 0.0))),
                        k4=float(entry.get("k4", getattr(self, "camera_k4", 0.0))),
                        distortion_model=str(
                            entry.get(
                                "distortion_model",
                                getattr(self, "camera_distortion_model", "plumb_bob"),
                            )
                        ),
                    )
                    ext = LidarToCameraExtrinsics.from_rpy_translation(
                        tx=float(entry.get("tx_m", entry.get("tx", self.lidar_to_camera_tx_m))),
                        ty=float(entry.get("ty_m", entry.get("ty", self.lidar_to_camera_ty_m))),
                        tz=float(entry.get("tz_m", entry.get("tz", self.lidar_to_camera_tz_m))),
                        roll=float(entry.get("roll_rad", entry.get("roll", self.lidar_to_camera_roll_rad))),
                        pitch=float(entry.get("pitch_rad", entry.get("pitch", self.lidar_to_camera_pitch_rad))),
                        yaw=float(entry.get("yaw_rad", entry.get("yaw", self.lidar_to_camera_yaw_rad))),
                    )
                    self.camera_profiles[cid] = CameraSensorProfile(
                        camera_id=cid,
                        topic=topic,
                        frame_id=str(entry.get("frame_id", cid)).strip() or cid,
                        projector=self._make_projector(intr, ext),
                        use_tf_extrinsics=bool(entry.get("use_tf_extrinsics", False)),
                        camera_info_topic=str(entry.get("camera_info_topic", "")).strip(),
                        depth_topic=str(entry.get("depth_topic", "")).strip(),
                    )
                    if cid == "primary":
                        self.projector = self.camera_profiles[cid].projector
            except Exception as exc:
                self.get_logger().warning(f"Invalid camera_profiles_json ignored: {exc}")

        self.multi_camera_fusion = MultiCameraEarlyFusion(self.camera_profiles)
        self.camera_buffers: Dict[str, Deque[Tuple[float, CameraConeDetections]]] = {}
        self.depth_buffers: Dict[str, Deque[Tuple[float, StereoDepthFrame]]] = {}
        for cid in self.camera_profiles:
            if cid == "primary":
                self.camera_buffers[cid] = self.camera_buffer
            else:
                self.camera_buffers[cid] = deque(maxlen=self.queue_size_camera)
            self.depth_buffers[cid] = deque(maxlen=max(4, min(16, self.queue_size_camera)))

        self.lidar_profiles: Dict[str, LidarSensorProfile] = {
            "primary": LidarSensorProfile(
                lidar_id="primary",
                topic=self.lidar_topic,
                frame_id=self.target_frame or "lidar",
            )
        }
        seen_lidar_topics = {self.lidar_topic}
        for idx, topic in enumerate(getattr(self, "lidar_topics_extra", []) or []):
            t = str(topic).strip()
            if not t or t in seen_lidar_topics:
                continue
            seen_lidar_topics.add(t)
            lid_id = f"lidar_extra_{idx}"
            self.lidar_profiles[lid_id] = LidarSensorProfile(
                lidar_id=lid_id, topic=t, frame_id=lid_id
            )

        lidar_json = getattr(self, "lidar_profiles_json", "").strip()
        if lidar_json:
            try:
                parsed = json.loads(lidar_json)
                if isinstance(parsed, dict):
                    entries = [
                        {"lidar_id": k, **v} if isinstance(v, dict) else {"lidar_id": k}
                        for k, v in parsed.items()
                    ]
                elif isinstance(parsed, list):
                    entries = parsed
                else:
                    entries = []
                for idx, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    lid_id = str(entry.get("lidar_id", f"lidar_{idx}")).strip() or f"lidar_{idx}"
                    topic = str(entry.get("topic", "")).strip()
                    if not topic:
                        continue
                    self.lidar_profiles[lid_id] = LidarSensorProfile(
                        lidar_id=lid_id,
                        topic=topic,
                        frame_id=str(entry.get("frame_id", lid_id)).strip() or lid_id,
                        offset_x_m=float(entry.get("offset_x_m", entry.get("tx_m", 0.0))),
                        offset_y_m=float(entry.get("offset_y_m", entry.get("ty_m", 0.0))),
                        offset_z_m=float(entry.get("offset_z_m", entry.get("tz_m", 0.0))),
                        yaw_rad=float(entry.get("yaw_rad", entry.get("yaw", 0.0))),
                    )
            except Exception as exc:
                self.get_logger().warning(f"Invalid lidar_profiles_json ignored: {exc}")

    def _on_camera(self, msg: CameraConeDetections, camera_id: str = "primary") -> None:
        stamp = self._stamp_to_sec(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp + self.max_rewind_s < self.last_processed_lidar_stamp:
            self.counters["oosm_dropped"] += 1
            return
        if stamp < self.last_processed_lidar_stamp:
            self.counters["oosm_within_budget"] += 1
            self._request_replay(stamp)
        cam_buffers = getattr(self, "camera_buffers", None)
        target_buf = (
            cam_buffers[camera_id]
            if cam_buffers is not None and camera_id in cam_buffers
            else self.camera_buffer
        )
        self._push_sorted(target_buf, (stamp, msg), "camera_overflow")

    def _on_depth(self, msg: Image, camera_id: str = "primary") -> None:
        if not getattr(self, "enable_stereo_depth_fusion", True):
            return
        stamp = self._stamp_to_sec(msg.header.stamp.sec, msg.header.stamp.nanosec)
        enc = str(getattr(msg, "encoding", "")).strip()
        enc_lower = enc.lower()
        if enc_lower not in ("32fc1", "16uc1", "32fc1_disparity", "disparity"):
            return
        scale = float(getattr(self, "stereo_depth_scale", 0.0))
        if scale <= 0.0:
            scale = 0.001 if enc_lower == "16uc1" else 1.0
        frame = StereoDepthFrame(
            stamp=stamp,
            width=int(getattr(msg, "width", 0)),
            height=int(getattr(msg, "height", 0)),
            encoding=enc,
            data=bytes(getattr(msg, "data", b"")),
            row_step_bytes=int(getattr(msg, "step", 0)),
            depth_scale=scale,
            baseline_m=float(getattr(self, "stereo_baseline_m", 0.12)),
            disparity_sigma_px=float(getattr(self, "stereo_disparity_sigma_px", 0.5)),
            min_depth_m=float(getattr(self, "stereo_depth_min_m", 0.5)),
            max_depth_m=float(getattr(self, "stereo_depth_max_m", 45.0)),
        )
        depth_buffers = getattr(self, "depth_buffers", None)
        if depth_buffers is None:
            self.depth_buffers = {"primary": deque(maxlen=8)}
            depth_buffers = self.depth_buffers
        if camera_id not in depth_buffers:
            depth_buffers[camera_id] = deque(maxlen=8)
        buf = depth_buffers[camera_id]
        data = list(buf)
        data.append((stamp, frame))
        data.sort(key=lambda v: v[0])
        if buf.maxlen and len(data) > buf.maxlen:
            data = data[-buf.maxlen :]
        buf.clear()
        buf.extend(data)
        if "stereo_depth_frames" in self.counters:
            self.counters["stereo_depth_frames"] += 1

    def _nearest_depth_for_id(
        self, reference_stamp: float, camera_id: str = "primary"
    ) -> Optional[StereoDepthFrame]:
        if not getattr(self, "enable_stereo_depth_fusion", True):
            return None
        depth_buffers = getattr(self, "depth_buffers", None)
        if not depth_buffers:
            return None
        buf = depth_buffers.get(camera_id)
        if not buf:
            return None
        max_age_s = float(getattr(self, "stereo_depth_max_age_ms", 150.0)) * 1e-3
        best_stamp, best_frame = min(buf, key=lambda pair: abs(pair[0] - reference_stamp))
        if abs(best_stamp - reference_stamp) > max_age_s:
            return None
        return best_frame

    def _on_extra_lidar(self, msg: LidarConeClusters, lidar_id: str) -> None:
        profiles = getattr(self, "lidar_profiles", {})
        profile = profiles.get(lidar_id)
        if "extra_lidar_messages" in self.counters:
            self.counters["extra_lidar_messages"] += 1
        if profile is None or (
            abs(profile.offset_x_m) < 1e-9
            and abs(profile.offset_y_m) < 1e-9
            and abs(profile.offset_z_m) < 1e-9
            and abs(profile.yaw_rad) < 1e-9
        ):
            self._on_lidar(msg)
            return

        c_y = math.cos(profile.yaw_rad)
        s_y = math.sin(profile.yaw_rad)
        transformed = LidarConeClusters()
        if transformed.header.stamp is not None and hasattr(msg.header.stamp, "sec"):
            transformed.header.stamp.sec = int(msg.header.stamp.sec)
            transformed.header.stamp.nanosec = int(msg.header.stamp.nanosec)
        else:
            transformed.header.stamp = msg.header.stamp
        transformed.header.frame_id = self.target_frame or msg.header.frame_id
        if hasattr(transformed, "lidar_quality") and hasattr(msg, "lidar_quality"):
            transformed.lidar_quality = float(msg.lidar_quality)
        if hasattr(transformed, "sensor_health") and hasattr(msg, "sensor_health"):
            transformed.sensor_health = msg.sensor_health
        out_clusters = []
        for c in msg.clusters:
            x = float(c.centroid.x)
            y = float(c.centroid.y)
            z = float(c.centroid.z)
            nc = LidarConeCluster()
            nc.centroid = Point(
                x=c_y * x - s_y * y + profile.offset_x_m,
                y=s_y * x + c_y * y + profile.offset_y_m,
                z=z + profile.offset_z_m,
            )
            cov = list(c.covariance) if len(c.covariance) == 9 else [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25]
            if abs(profile.yaw_rad) >= 1e-9:
                c00, c01, c02 = float(cov[0]), float(cov[1]), float(cov[2])
                c10, c11, c12 = float(cov[3]), float(cov[4]), float(cov[5])
                c20, c21, c22 = float(cov[6]), float(cov[7]), float(cov[8])
                t00 = c_y * c00 - s_y * c10
                t01 = c_y * c01 - s_y * c11
                t10 = s_y * c00 + c_y * c10
                t11 = s_y * c01 + c_y * c11
                nc.covariance = [
                    t00 * c_y - t01 * s_y,
                    t00 * s_y + t01 * c_y,
                    c_y * c02 - s_y * c12,
                    t10 * c_y - t11 * s_y,
                    t10 * s_y + t11 * c_y,
                    s_y * c02 + c_y * c12,
                    c20 * c_y - c21 * s_y,
                    c20 * s_y + c21 * c_y,
                    c22,
                ]
            else:
                nc.covariance = cov
            if hasattr(c, "point_count"):
                nc.point_count = c.point_count
            if hasattr(c, "geometry_quality"):
                nc.geometry_quality = c.geometry_quality
            if hasattr(c, "stripe_class_probs"):
                nc.stripe_class_probs = list(c.stripe_class_probs)
            out_clusters.append(nc)
        transformed.clusters = out_clusters
        self._on_lidar(transformed)

    def _on_lidar(self, msg: LidarConeClusters) -> None:
        stamp = self._stamp_to_sec(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp + self.max_rewind_s < self.last_processed_lidar_stamp:
            self.counters["oosm_dropped"] += 1
            return
        if stamp < self.last_processed_lidar_stamp:
            self.counters["oosm_within_budget"] += 1
            self._request_replay(stamp)
        self._push_sorted(self.lidar_buffer, (stamp, msg), "lidar_overflow")

    @staticmethod
    def _extract_twist_from_motion_msg(msg: Any) -> Tuple[float, float, float]:
        """Extract (vx, vy, wz) from Odometry, TwistWithCovarianceStamped, or TwistStamped."""
        tw_outer = getattr(msg, "twist", None)
        tw = getattr(tw_outer, "twist", None)
        if tw is None:
            tw = tw_outer
        lin = getattr(tw, "linear", None)
        ang = getattr(tw, "angular", None)
        vx_v = float(getattr(lin, "x", 0.0)) if lin is not None else 0.0
        vy_v = float(getattr(lin, "y", 0.0)) if lin is not None else 0.0
        wz_v = float(getattr(ang, "z", 0.0)) if ang is not None else 0.0
        return vx_v, vy_v, wz_v

    def _on_motion(self, msg: Any) -> None:
        stamp = self._stamp_to_sec(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._push_sorted(self.motion_buffer, (stamp, msg), "motion_overflow")
        vx, vy, _wz = self._extract_twist_from_motion_msg(msg)
        tw_outer = getattr(msg, "twist", None)
        tw = getattr(tw_outer, "twist", None) or tw_outer
        lin = getattr(tw, "linear", None)
        vz = float(getattr(lin, "z", 0.0)) if lin is not None else 0.0
        self.last_ego_speed_mps = float((vx * vx + vy * vy + vz * vz) ** 0.5)
        pose_obj = getattr(getattr(msg, "pose", None), "pose", None)
        q = getattr(pose_obj, "orientation", None)
        if q is not None and hasattr(q, "w"):
            qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
            sinr_cosp = 2.0 * (qw * qx + qy * qz)
            cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
            self.last_ego_roll_rad = math.atan2(sinr_cosp, cosr_cosp)
            sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
            self.last_ego_pitch_rad = math.asin(sinp)

    def _request_replay(self, anchor_stamp: float) -> None:
        self.pending_replay = True
        self.counters["replay_requests"] += 1
        if self.replay_anchor_stamp is None:
            self.replay_anchor_stamp = anchor_stamp
        else:
            self.replay_anchor_stamp = min(self.replay_anchor_stamp, anchor_stamp)

    def _push_sorted(self, buffer: Deque, item: Tuple, counter_key: str) -> None:
        data = list(buffer)
        data.append(item)
        data.sort(key=lambda v: v[0])
        overflow = max(0, len(data) - buffer.maxlen)
        if overflow > 0:
            if self.drop_policy == "drop_oldest":
                data = data[overflow:]
            else:
                data = data[:-overflow]
            self.counters[counter_key] += overflow
        buffer.clear()
        buffer.extend(data)

    def _process_once(self) -> None:
        cycle_start_ns = time.perf_counter_ns()
        if not self.lidar_buffer:
            if self.enable_camera_fallback:
                self._process_camera_fallback_without_lidar()
            if self.pending_replay:
                self._run_replay_if_needed()
            self._record_stage_timing("cycle_total", self._elapsed_ms(cycle_start_ns))
            return

        lidar_stamp, lidar_msg = self.lidar_buffer.popleft()
        self._update_motion_alignment(lidar_stamp)
        if lidar_stamp + self.max_rewind_s < self.last_processed_lidar_stamp:
            self.counters["oosm_dropped"] += 1
            self._record_stage_timing("cycle_total", self._elapsed_ms(cycle_start_ns))
            return

        camera_lookup_start_ns = time.perf_counter_ns()
        camera_msg, camera_msgs_by_cam = self._lookup_active_cameras(lidar_stamp)
        self._record_stage_timing("camera_lookup", self._elapsed_ms(camera_lookup_start_ns))
        extrinsics_start_ns = time.perf_counter_ns()
        self._refresh_projector_extrinsics(
            lidar_msg=lidar_msg,
            camera_msg=camera_msg,
            camera_msgs_by_cam=camera_msgs_by_cam,
        )
        self._record_stage_timing("extrinsics_lookup", self._elapsed_ms(extrinsics_start_ns))

        is_oosm_lidar = lidar_stamp < self.last_processed_lidar_stamp
        source = "oosm_live" if is_oosm_lidar else "live"
        build_start_ns = time.perf_counter_ns()
        observations, tracker_observations = self._build_observations(
            lidar_msg=lidar_msg,
            camera_msg=camera_msg,
            source=source,
            camera_msgs_by_cam=camera_msgs_by_cam,
        )
        self._record_stage_timing("build_observations", self._elapsed_ms(build_start_ns))
        publish_start_ns = time.perf_counter_ns()
        self._publish_observation_messages(observations)
        self._record_stage_timing("publish_observations", self._elapsed_ms(publish_start_ns))
        self.last_obs_count = len(observations)

        if self.enable_tracks:
            if is_oosm_lidar:
                self._upsert_frame_record(
                    FrameRecord(
                        stamp=lidar_stamp,
                        lidar_msg=lidar_msg,
                        snapshot_before=None,
                        snapshot_after=None,
                    )
                )
                self._request_replay(lidar_stamp)
            else:
                tracker_start_ns = time.perf_counter_ns()
                snapshot_before = self.tracker.snapshot()
                self._apply_tracker_step(
                    observations=tracker_observations,
                    header=lidar_msg.header,
                    stamp=lidar_stamp,
                )
                self._record_stage_timing("tracker_step", self._elapsed_ms(tracker_start_ns))
                snapshot_after = self.tracker.snapshot()
                self._upsert_frame_record(
                    FrameRecord(
                        stamp=lidar_stamp,
                        lidar_msg=lidar_msg,
                        snapshot_before=snapshot_before,
                        snapshot_after=snapshot_after,
                    )
                )
        else:
            self._record_stage_timing("tracker_step", 0.0)

        if is_oosm_lidar:
            self._record_stage_timing("tracker_step", 0.0)
        if not self.enable_tracks:
            self._upsert_frame_record(
                FrameRecord(
                    stamp=lidar_stamp,
                    lidar_msg=lidar_msg,
                    snapshot_before=None,
                    snapshot_after=None,
                )
            )

        if not is_oosm_lidar:
            self.last_processed_lidar_stamp = max(self.last_processed_lidar_stamp, lidar_stamp)

        self._update_latency_and_blind_distance(lidar_stamp)

        if self.pending_replay:
            self._run_replay_if_needed()
        self._record_stage_timing("cycle_total", self._elapsed_ms(cycle_start_ns))

    def _run_replay_if_needed(self) -> None:
        replay_start_ns = time.perf_counter_ns()
        if not self.pending_replay:
            self._record_stage_timing("replay_total", 0.0)
            return
        self.pending_replay = False

        if not self.enable_tracks:
            self.replay_anchor_stamp = None
            self._record_stage_timing("replay_total", self._elapsed_ms(replay_start_ns))
            return
        if self.replay_anchor_stamp is None:
            self._record_stage_timing("replay_total", self._elapsed_ms(replay_start_ns))
            return

        anchor = self.replay_anchor_stamp
        self.replay_anchor_stamp = None

        replay_frames = [f for f in self.frame_history if f.stamp >= anchor]
        if not replay_frames:
            self.counters["replay_no_frames"] += 1
            self._record_stage_timing("replay_total", self._elapsed_ms(replay_start_ns))
            return
        if len(replay_frames) > self.max_replay_steps:
            self.counters["replay_budget_exceeded"] += 1
            self._record_stage_timing("replay_total", self._elapsed_ms(replay_start_ns))
            return

        base_snapshot = self._resolve_replay_base_snapshot(replay_frames[0])
        if base_snapshot is None:
            self.counters["replay_no_base_snapshot"] += 1
            self._record_stage_timing("replay_total", self._elapsed_ms(replay_start_ns))
            return

        self.tracker.restore(base_snapshot)
        replayed = 0

        for frame in replay_frames:
            self._update_motion_alignment(frame.stamp)
            camera_msg, camera_msgs_by_cam = self._lookup_active_cameras(frame.stamp)
            self._refresh_projector_extrinsics(
                lidar_msg=frame.lidar_msg,
                camera_msg=camera_msg,
                camera_msgs_by_cam=camera_msgs_by_cam,
            )

            observations, tracker_observations = self._build_observations(
                lidar_msg=frame.lidar_msg,
                camera_msg=camera_msg,
                source="replay",
                camera_msgs_by_cam=camera_msgs_by_cam,
            )
            self._publish_observation_messages(observations)
            snapshot_before = self.tracker.snapshot()
            self._apply_tracker_step(
                observations=tracker_observations,
                header=frame.lidar_msg.header,
                stamp=frame.stamp,
            )
            snapshot_after = self.tracker.snapshot()
            self._upsert_frame_record(
                FrameRecord(
                    stamp=frame.stamp,
                    lidar_msg=frame.lidar_msg,
                    snapshot_before=snapshot_before,
                    snapshot_after=snapshot_after,
                )
            )
            replayed += 1

        self.counters["replay_completed"] += 1
        self.counters["replay_frames"] += replayed
        self.last_processed_lidar_stamp = max(
            self.last_processed_lidar_stamp, replay_frames[-1].stamp
        )
        self._record_stage_timing("replay_total", self._elapsed_ms(replay_start_ns))

    def _resolve_replay_base_snapshot(
        self, first_replay_frame: FrameRecord
    ) -> Optional[TrackerSnapshot]:
        if first_replay_frame.snapshot_before is not None:
            return first_replay_frame.snapshot_before

        prev_candidates = [
            frame
            for frame in self.frame_history
            if frame.stamp < first_replay_frame.stamp and frame.snapshot_after is not None
        ]
        if prev_candidates:
            return prev_candidates[-1].snapshot_after

        if self.history_base_snapshot is not None:
            return self.history_base_snapshot
        return None

    def _upsert_frame_record(self, new_record: FrameRecord) -> None:
        replaced = False
        for idx, existing in enumerate(self.frame_history):
            if abs(existing.stamp - new_record.stamp) <= 1e-9:
                self.frame_history[idx] = new_record
                replaced = True
                break
            if existing.stamp > new_record.stamp:
                self.frame_history.insert(idx, new_record)
                replaced = True
                break

        if not replaced:
            self.frame_history.append(new_record)

        while len(self.frame_history) > max(1, self.frame_history_size):
            removed = self.frame_history.pop(0)
            if removed.snapshot_after is not None:
                self.history_base_snapshot = removed.snapshot_after

    def _nearest_camera(
        self, reference_stamp: float
    ) -> Optional[Tuple[float, CameraConeDetections]]:
        while self.camera_buffer:
            oldest_stamp = self.camera_buffer[0][0]
            if reference_stamp - oldest_stamp > self.camera_retention_s:
                self.camera_buffer.popleft()
            else:
                break

        if not self.camera_buffer:
            return None
        return min(self.camera_buffer, key=lambda pair: abs(pair[0] - reference_stamp))

    def _nearest_camera_for_id(
        self, reference_stamp: float, camera_id: str = "primary"
    ) -> Optional[Tuple[float, CameraConeDetections]]:
        if camera_id == "primary":
            return self._nearest_camera(reference_stamp)
        cam_buffers = getattr(self, "camera_buffers", {})
        buf = cam_buffers.get(camera_id)
        if buf is None:
            return None
        while buf:
            oldest_stamp = buf[0][0]
            if reference_stamp - oldest_stamp > self.camera_retention_s:
                buf.popleft()
            else:
                break
        if not buf:
            return None
        return min(buf, key=lambda pair: abs(pair[0] - reference_stamp))

    def _lookup_active_cameras(
        self, lidar_stamp: float
    ) -> Tuple[Optional[CameraConeDetections], Dict[str, CameraConeDetections]]:
        profiles = getattr(self, "camera_profiles", None)
        if not profiles or len(profiles) <= 1:
            camera_msg = None
            nearest_camera = self._nearest_camera(lidar_stamp)
            if nearest_camera is not None:
                cam_stamp, cam_msg = nearest_camera
                self._update_camera_lidar_sync(cam_stamp, lidar_stamp)
                if abs(cam_stamp - lidar_stamp) <= self.max_camera_lidar_delta_s:
                    camera_msg = cam_msg
                else:
                    self.counters["camera_miss"] += 1
            by_cam = {"primary": camera_msg} if camera_msg is not None else {}
            return camera_msg, by_cam

        by_cam: Dict[str, CameraConeDetections] = {}
        primary_msg: Optional[CameraConeDetections] = None
        primary_synced = False
        for cam_id in profiles:
            nearest = self._nearest_camera_for_id(lidar_stamp, cam_id)
            if nearest is None:
                continue
            cam_stamp, cam_msg = nearest
            if cam_id == "primary":
                self._update_camera_lidar_sync(cam_stamp, lidar_stamp)
                primary_synced = True
            if abs(cam_stamp - lidar_stamp) <= self.max_camera_lidar_delta_s:
                by_cam[cam_id] = cam_msg
                if cam_id == "primary":
                    primary_msg = cam_msg

        if not primary_synced and by_cam:
            first_msg = next(iter(by_cam.values()))
            cam_stamp = self._stamp_to_sec(
                first_msg.header.stamp.sec, first_msg.header.stamp.nanosec
            )
            self._update_camera_lidar_sync(cam_stamp, lidar_stamp)
        if not by_cam and any(
            getattr(self, "camera_buffers", {}).get(cid) for cid in profiles
        ):
            self.counters["camera_miss"] += 1
        if primary_msg is None and by_cam:
            primary_msg = next(iter(by_cam.values()))
        return primary_msg, by_cam

    def _build_observations(
        self,
        lidar_msg: LidarConeClusters,
        camera_msg: Optional[CameraConeDetections],
        source: str,
        camera_msgs_by_cam: Optional[Dict[str, CameraConeDetections]] = None,
    ) -> Tuple[List[ConeObservation], List[Observation2D]]:
        fallback_class_probs = (
            [0.25, 0.25, 0.25, 0.25]
            if self.enable_projection_semantics
            else self._aggregate_class_probs(camera_msg)
        )
        fallback_semantic_weight = self._semantic_weight(camera_msg)
        health = self._health_code(camera_msg, lidar_msg)
        self.last_health = health
        detections = []
        detections_by_cam: Dict[str, List[Any]] = {}
        cam_stamps_by_cam: Dict[str, float] = {}
        lidar_stamp_s = self._stamp_to_sec(
            lidar_msg.header.stamp.sec, lidar_msg.header.stamp.nanosec
        )
        cam_stamp_s = (
            self._stamp_to_sec(
                camera_msg.header.stamp.sec, camera_msg.header.stamp.nanosec
            )
            if camera_msg is not None and hasattr(camera_msg, "header")
            else lidar_stamp_s
        )
        use_att = bool(getattr(self, "enable_chassis_attitude_compensation", True))
        ego_speed = getattr(self, "last_ego_speed_mps", 0.0)
        ego_pitch = getattr(self, "last_ego_pitch_rad", 0.0) if use_att else 0.0
        ego_roll = getattr(self, "last_ego_roll_rad", 0.0) if use_att else 0.0

        multi_cam_fusion = getattr(self, "multi_camera_fusion", None)
        use_multi_cam = (
            self.enable_projection_semantics
            and multi_cam_fusion is not None
            and len(getattr(multi_cam_fusion, "profiles", {})) > 1
            and bool(camera_msgs_by_cam)
        )

        if use_multi_cam and camera_msgs_by_cam is not None:
            for cam_id, cam_msg_item in camera_msgs_by_cam.items():
                prof = multi_cam_fusion.profiles.get(cam_id)
                if prof is None:
                    continue
                prof.projector.apply_aero_compensation(
                    ego_speed,
                    chassis_pitch_rad=ego_pitch,
                    chassis_roll_rad=ego_roll,
                )
                detections_by_cam[cam_id] = prof.projector.detections_from_msg(cam_msg_item)
                cam_stamps_by_cam[cam_id] = (
                    self._stamp_to_sec(
                        cam_msg_item.header.stamp.sec, cam_msg_item.header.stamp.nanosec
                    )
                    if hasattr(cam_msg_item, "header")
                    else lidar_stamp_s
                )
            detections = detections_by_cam.get("primary", [])
            if not detections and detections_by_cam:
                detections = next(iter(detections_by_cam.values()))
        elif self.enable_projection_semantics and camera_msg is not None:
            self.projector.apply_aero_compensation(
                ego_speed,
                chassis_pitch_rad=ego_pitch,
                chassis_roll_rad=ego_roll,
            )
            detections = self.projector.detections_from_msg(camera_msg)

        frame_id = self.target_frame if self.target_frame else lidar_msg.header.frame_id
        observations: List[ConeObservation] = []
        tracker_observations: List[Observation2D] = []
        frame_projection_residual_sum = 0.0
        frame_projection_residual_samples = 0
        frame_projection_residual_max = 0.0
        matched_detection_indices: List[int] = []
        cluster_positions: List[Tuple[float, float, float]] = []

        min_uncorr_pts = int(getattr(self, "min_uncorroborated_cluster_points", 2))
        req_cam_single = bool(getattr(self, "require_camera_for_single_point_clusters", True))
        assume_uncounted = bool(getattr(self, "assume_uncounted_clusters_valid", True))

        sorted_clusters = sorted(
            lidar_msg.clusters,
            key=lambda c: float(c.centroid.x) ** 2 + float(c.centroid.y) ** 2,
        )

        batch_estimates = None
        if use_multi_cam and sorted_clusters and any(detections_by_cam.values()):
            clusters_lidar_by_cam: Dict[
                str, List[Tuple[Tuple[float, float, float], List[float]]]
            ] = {}
            for cam_id, cam_dets in detections_by_cam.items():
                if not cam_dets:
                    continue
                target_cam_stamp = cam_stamps_by_cam.get(cam_id, lidar_stamp_s)
                clusters_lidar_by_cam[cam_id] = [
                    (
                        self._transform_point_between_stamps(
                            (
                                float(c.centroid.x),
                                float(c.centroid.y),
                                float(c.centroid.z),
                            ),
                            from_stamp=lidar_stamp_s,
                            to_stamp=target_cam_stamp,
                        ),
                        list(c.covariance),
                    )
                    for c in sorted_clusters
                ]
            if clusters_lidar_by_cam:
                batch_estimates = multi_cam_fusion.estimate_batch_multi_camera_semantics(
                    clusters_lidar_by_cam=clusters_lidar_by_cam,
                    detections_by_cam=detections_by_cam,
                    fallback_probs=fallback_class_probs,
                )
        elif (
            self.enable_projection_semantics
            and camera_msg is not None
            and getattr(self, "enable_hungarian_early_fusion", False)
            and len(sorted_clusters) > 1
            and len(detections) > 1
        ):
            clusters_at_cam = [
                (
                    self._transform_point_between_stamps(
                        (
                            float(c.centroid.x),
                            float(c.centroid.y),
                            float(c.centroid.z),
                        ),
                        from_stamp=lidar_stamp_s,
                        to_stamp=cam_stamp_s,
                    ),
                    list(c.covariance),
                )
                for c in sorted_clusters
            ]
            batch_estimates = self.projector.estimate_batch_cluster_semantics(
                clusters_lidar=clusters_at_cam,
                detections=detections,
                fallback_probs=fallback_class_probs,
            )

        active_cam_present = (camera_msg is not None) or (
            use_multi_cam and any(detections_by_cam.values())
        )

        for c_idx, cluster in enumerate(sorted_clusters):
            c_pos = (
                float(cluster.centroid.x),
                float(cluster.centroid.y),
                float(cluster.centroid.z),
            )
            raw_pt_cnt = getattr(cluster, "point_count", None)
            if raw_pt_cnt is None or int(raw_pt_cnt) <= 0:
                pt_cnt = min_uncorr_pts if assume_uncounted else 0
            else:
                pt_cnt = int(raw_pt_cnt)
            class_probs = fallback_class_probs
            semantic_weight = fallback_semantic_weight
            cluster_residual_mean_px = 0.0
            matched_det_idx = None
            estimate = None
            if self.enable_projection_semantics and active_cam_present:
                if batch_estimates is not None:
                    estimate = batch_estimates[c_idx]
                elif camera_msg is not None:
                    c_pos_at_cam = self._transform_point_between_stamps(
                        c_pos, from_stamp=lidar_stamp_s, to_stamp=cam_stamp_s
                    )
                    estimate = self.projector.estimate_cluster_semantics(
                        centroid_lidar=c_pos_at_cam,
                        covariance=list(cluster.covariance),
                        detections=detections,
                        fallback_probs=fallback_class_probs,
                        excluded_detection_indices=matched_detection_indices,
                    )
                if estimate is not None:
                    class_probs = [float(v) for v in estimate.class_probs]
                    semantic_weight = float(estimate.semantic_weight)
                    matched_det_idx = estimate.diagnostics.matched_detection_idx
                    if matched_det_idx is not None:
                        matched_detection_indices.append(int(matched_det_idx))
            if (
                matched_det_idx is None
                and getattr(self, "enable_lidar_stripe_semantics", True)
            ):
                stripe_probs = getattr(cluster, "stripe_class_probs", None)
                if (
                    stripe_probs is not None
                    and len(stripe_probs) == 4
                    and max(stripe_probs) > 0.35
                ):
                    class_probs = [float(v) for v in stripe_probs]
            if req_cam_single and pt_cnt < min_uncorr_pts and matched_det_idx is None:
                continue
            cluster_positions.append(c_pos)
            if self.enable_projection_semantics and estimate is not None:
                self.counters["projection_projected_samples"] += int(
                    estimate.diagnostics.projected_samples
                )
                self.counters["projection_matched_samples"] += int(
                    estimate.diagnostics.matched_samples
                )
                if (
                    estimate.diagnostics.projected_samples > 0
                    and estimate.diagnostics.matched_samples == 0
                ):
                    self.counters["projection_clusters_no_match"] += 1
                if estimate.diagnostics.residual_samples > 0:
                    cluster_residual_mean_px = (
                        estimate.diagnostics.residual_sum_px
                        / float(estimate.diagnostics.residual_samples)
                    )
                    frame_projection_residual_sum += float(estimate.diagnostics.residual_sum_px)
                    frame_projection_residual_samples += int(estimate.diagnostics.residual_samples)
                    frame_projection_residual_max = max(
                        frame_projection_residual_max,
                        float(estimate.diagnostics.residual_max_px),
                    )
                    self.projection_residual_sum_px_total += float(
                        estimate.diagnostics.residual_sum_px
                    )
                    self.projection_residual_samples_total += int(
                        estimate.diagnostics.residual_samples
                    )
                    self.projection_residual_max_px = max(
                        self.projection_residual_max_px,
                        float(estimate.diagnostics.residual_max_px),
                    )
                    semantic_weight *= self._projection_residual_penalty(
                        cluster_residual_mean_px
                    )
                    if cluster_residual_mean_px >= self.projection_residual_warn_px:
                        self.counters["projection_high_residual_clusters"] += 1

            obs = ConeObservation()
            obs.header.stamp = lidar_msg.header.stamp
            obs.header.frame_id = frame_id
            obs.position = cluster.centroid
            obs.covariance = self._scaled_covariance(
                covariance=list(cluster.covariance),
                health=health,
                projection_residual_mean_px=cluster_residual_mean_px,
                camera_lidar_abs_delta_ms=self.last_camera_lidar_abs_delta_ms,
                motion_age_ms=self.last_motion_age_ms,
                motion_comp_error_m=self.last_motion_comp_error_m,
            )
            obs.class_probs = class_probs

            raw_geom_q = getattr(cluster, "geometry_quality", None)
            if raw_geom_q is None or float(raw_geom_q) <= 0.0:
                geom_q = 1.0
            else:
                geom_q = float(raw_geom_q)
            quality = geom_q * semantic_weight
            if health == SENSOR_HEALTH_DEGRADED:
                quality *= 0.8
            elif health == SENSOR_HEALTH_CRITICAL:
                quality *= 0.6
            obs.quality_score = float(max(0.0, min(1.0, quality)))
            obs.sensor_health = health
            obs.source = source
            observations.append(obs)

            if getattr(self, "enable_tracks", True):
                tracker_observations.append(self._to_tracker_observation(obs))

        if cluster_positions and self.enable_projection_semantics:
            sanity = self.projector.check_extrinsics_sanity(cluster_positions)
            self.last_extrinsics_sanity = sanity
            if sanity.startswith("error_") or sanity.startswith("warn_"):
                streak = int(getattr(self, "_extrinsics_sanity_warn_streak", 0)) + 1
                self._extrinsics_sanity_warn_streak = streak
                if streak in (5, 50) or (streak > 50 and streak % 100 == 0):
                    logger = self.get_logger() if hasattr(self, "get_logger") else None
                    if logger is not None:
                        logger.warn(
                            f"Extrinsics/frame sanity check reported '{sanity}' "
                            f"(streak={streak}). Check camera optical frame vs. body frame or TF extrinsics."
                        )
            else:
                self._extrinsics_sanity_warn_streak = 0

        # Bidirectional Camera -> 3D Frustum recovery for unmatched distant cones
        if (
            getattr(self, "enable_bidirectional_frustum_fusion", False)
            and self.enable_projection_semantics
            and active_cam_present
        ):
            enable_gnd_depth = bool(getattr(self, "enable_ground_plane_depth_fusion", True))
            gnd_z = float(getattr(self, "ground_z_base_m", -0.22))
            gnd_slope = float(getattr(self, "ground_slope_rad", 0.0))
            enable_stereo = bool(getattr(self, "enable_stereo_depth_fusion", True))

            if use_multi_cam and camera_msgs_by_cam is not None:
                stereo_by_cam: Dict[str, StereoDepthFrame] = {}
                for cid in detections_by_cam:
                    c_st = cam_stamps_by_cam.get(cid, lidar_stamp_s)
                    d_fr = self._nearest_depth_for_id(c_st, camera_id=cid)
                    if d_fr is not None:
                        stereo_by_cam[cid] = d_fr
                recovered_by_cam = multi_cam_fusion.recover_multi_camera_unmatched_detections(
                    detections_by_cam=detections_by_cam,
                    matched_indices_by_cam=multi_cam_fusion.last_matched_indices_by_cam,
                    existing_positions=cluster_positions,
                    cone_height_m=getattr(self, "camera_fallback_cone_height_m", 0.30),
                    min_confidence=getattr(self, "frustum_min_confidence", 0.55),
                    min_bbox_height_px=getattr(self, "camera_fallback_min_bbox_height_px", 8.0),
                    min_depth_m=getattr(self, "camera_fallback_min_depth_m", 2.0),
                    max_depth_m=getattr(self, "camera_fallback_max_depth_m", 35.0),
                    dedup_radius_m=getattr(self, "frustum_dedup_radius_m", 1.2),
                    enable_ground_plane_depth_fusion=enable_gnd_depth,
                    ground_z_base_m=gnd_z,
                    ground_slope_rad=gnd_slope,
                    stereo_depth_by_cam=stereo_by_cam,
                    enable_stereo_depth_fusion=enable_stereo,
                )
                for cid, rec_list in recovered_by_cam.items():
                    c_stamp_item = cam_stamps_by_cam.get(cid, lidar_stamp_s)
                    c_msg_ref = camera_msgs_by_cam.get(cid, camera_msg)
                    img_q = float(getattr(c_msg_ref, "image_quality", 1.0))
                    for rec in rec_list:
                        rec_pos_lidar = self._transform_point_between_stamps(
                            rec.position, from_stamp=c_stamp_item, to_stamp=lidar_stamp_s
                        )
                        obs = ConeObservation()
                        obs.header.stamp = lidar_msg.header.stamp
                        obs.header.frame_id = frame_id
                        obs.position = Point(
                            x=float(rec_pos_lidar[0]),
                            y=float(rec_pos_lidar[1]),
                            z=float(rec_pos_lidar[2]),
                        )
                        obs.covariance = self._scaled_covariance(
                            covariance=rec.covariance,
                            health=health,
                            projection_residual_mean_px=0.0,
                            camera_lidar_abs_delta_ms=self.last_camera_lidar_abs_delta_ms,
                            motion_age_ms=self.last_motion_age_ms,
                            motion_comp_error_m=self.last_motion_comp_error_m,
                        )
                        obs.class_probs = [float(v) for v in rec.class_probs]
                        obs.quality_score = float(
                            max(0.1, min(1.0, rec.confidence * img_q * (0.82 if rec.used_stereo else 0.70)))
                        )
                        obs.sensor_health = health
                        obs.source = f"{source}_stereo_recovered" if rec.used_stereo else f"{source}_frustum_recovered"
                        observations.append(obs)
                        if "frustum_recovered_observations" in self.counters:
                            self.counters["frustum_recovered_observations"] += 1
                        if rec.used_stereo and "stereo_depth_fused_observations" in self.counters:
                            self.counters["stereo_depth_fused_observations"] += 1
                        if getattr(self, "enable_tracks", True):
                            tracker_observations.append(self._to_tracker_observation(obs))
            elif detections and camera_msg is not None:
                stereo_frame = self._nearest_depth_for_id(cam_stamp_s, camera_id="primary")
                recovered_cones = self.projector.recover_unmatched_detections(
                    detections=detections,
                    matched_detection_indices=matched_detection_indices,
                    existing_positions=cluster_positions,
                    cone_height_m=getattr(self, "camera_fallback_cone_height_m", 0.30),
                    min_confidence=getattr(self, "frustum_min_confidence", 0.55),
                    min_bbox_height_px=getattr(self, "camera_fallback_min_bbox_height_px", 8.0),
                    min_depth_m=getattr(self, "camera_fallback_min_depth_m", 2.0),
                    max_depth_m=getattr(self, "camera_fallback_max_depth_m", 35.0),
                    dedup_radius_m=getattr(self, "frustum_dedup_radius_m", 1.2),
                    enable_ground_plane_depth_fusion=enable_gnd_depth,
                    ground_z_base_m=gnd_z,
                    ground_slope_rad=gnd_slope,
                    stereo_depth=stereo_frame,
                    enable_stereo_depth_fusion=enable_stereo,
                )
                img_q = float(getattr(camera_msg, "image_quality", 1.0))
                for rec in recovered_cones:
                    rec_pos_lidar = self._transform_point_between_stamps(
                        rec.position, from_stamp=cam_stamp_s, to_stamp=lidar_stamp_s
                    )
                    obs = ConeObservation()
                    obs.header.stamp = lidar_msg.header.stamp
                    obs.header.frame_id = frame_id
                    obs.position = Point(
                        x=float(rec_pos_lidar[0]),
                        y=float(rec_pos_lidar[1]),
                        z=float(rec_pos_lidar[2]),
                    )
                    obs.covariance = self._scaled_covariance(
                        covariance=rec.covariance,
                        health=health,
                        projection_residual_mean_px=0.0,
                        camera_lidar_abs_delta_ms=self.last_camera_lidar_abs_delta_ms,
                        motion_age_ms=self.last_motion_age_ms,
                        motion_comp_error_m=self.last_motion_comp_error_m,
                    )
                    obs.class_probs = [float(v) for v in rec.class_probs]
                    obs.quality_score = float(
                        max(0.1, min(1.0, rec.confidence * img_q * (0.82 if rec.used_stereo else 0.70)))
                    )
                    obs.sensor_health = health
                    obs.source = f"{source}_stereo_recovered" if rec.used_stereo else f"{source}_frustum_recovered"
                    observations.append(obs)
                    if "frustum_recovered_observations" in self.counters:
                        self.counters["frustum_recovered_observations"] += 1
                    if rec.used_stereo and "stereo_depth_fused_observations" in self.counters:
                        self.counters["stereo_depth_fused_observations"] += 1
                    if getattr(self, "enable_tracks", True):
                        tracker_observations.append(self._to_tracker_observation(obs))

        if (
            getattr(self, "enable_camera_fallback", True)
            and not observations
            and camera_msg is not None
            and camera_msg.detections
        ):
            fallback_obs, fallback_tracker = self._build_camera_fallback_observations(
                camera_msg=camera_msg,
                stamp_sec=self._stamp_to_sec(
                    lidar_msg.header.stamp.sec, lidar_msg.header.stamp.nanosec
                ),
                source=f"{source}_camera_fallback",
            )
            observations.extend(fallback_obs)
            tracker_observations.extend(fallback_tracker)

        if frame_projection_residual_samples > 0:
            self.last_projection_residual_mean_px = float(
                frame_projection_residual_sum / float(frame_projection_residual_samples)
            )
        elif not self.enable_projection_semantics or not active_cam_present:
            self.last_projection_residual_mean_px = 0.0

        # Debug 2D reprojection annotations & online calibration window collection
        if (
            getattr(self, "enable_debug_viz", False)
            and camera_msg is not None
            and detections
            and cluster_positions
        ):
            clusters_at_cam_pos = [
                self._transform_point_between_stamps(
                    pos, from_stamp=lidar_stamp_s, to_stamp=cam_stamp_s
                )
                for pos in cluster_positions
            ]
            self._publish_reprojection_debug(
                clusters_xyz=clusters_at_cam_pos,
                detections=detections,
                header=lidar_msg.header,
            )

        if (
            getattr(self, "enable_online_calibration", False)
            and source in ("live", "oosm_live")
            and camera_msg is not None
            and detections
            and cluster_positions
        ):
            calib_window = getattr(self, "calibration_window", None)
            if calib_window is None:
                self.calibration_window = deque(
                    maxlen=max(5, int(getattr(self, "online_calibration_window_frames", 30)))
                )
                calib_window = self.calibration_window
            clusters_at_cam_pos = [
                self._transform_point_between_stamps(
                    pos, from_stamp=lidar_stamp_s, to_stamp=cam_stamp_s
                )
                for pos in cluster_positions
            ]
            vx_mps = float(getattr(self, "last_ego_speed_mps", 0.0))
            vy_mps = 0.0
            yaw_rate = 0.0
            motion_buf = getattr(self, "motion_buffer", None)
            if motion_buf:
                _, last_motion = motion_buf[-1]
                vx_mps, vy_mps, yaw_rate = self._extract_twist_from_motion_msg(last_motion)
            calib_window.append(
                CalibrationFrameSample(
                    clusters_xyz=clusters_at_cam_pos,
                    detections=detections,
                    vx_mps=vx_mps,
                    vy_mps=vy_mps,
                    yaw_rate_radps=yaw_rate,
                )
            )
            self._online_calib_frames_since_run = (
                int(getattr(self, "_online_calib_frames_since_run", 0)) + 1
            )
            win_target = max(5, int(getattr(self, "online_calibration_window_frames", 30)))
            if (
                len(calib_window) >= min(win_target, calib_window.maxlen or win_target)
                and self._online_calib_frames_since_run >= win_target
            ):
                self._online_calib_frames_since_run = 0
                self.run_online_calibration()

        return observations, tracker_observations

    def _publish_observation_messages(self, observations: List[ConeObservation]) -> None:
        for obs in observations:
            self.obs_pub.publish(obs)

    def _to_tracker_observation(self, obs: ConeObservation) -> Observation2D:
        covariance = list(obs.covariance) if len(obs.covariance) == 9 else [0.25] * 9
        cov_xx = float(max(1e-6, covariance[0]))
        cov_xy = float(0.5 * (covariance[1] + covariance[3]))
        cov_yy = float(max(1e-6, covariance[4]))
        z_var = float(max(1e-6, covariance[8]))
        return Observation2D(
            x=float(obs.position.x),
            y=float(obs.position.y),
            z=float(obs.position.z),
            cov_xx=cov_xx,
            cov_xy=cov_xy,
            cov_yy=cov_yy,
            z_var=z_var,
            class_probs=[float(v) for v in obs.class_probs],
            quality=float(obs.quality_score),
        )

    def _transform_point_between_stamps(
        self,
        point: Tuple[float, float, float],
        from_stamp: float,
        to_stamp: float,
    ) -> Tuple[float, float, float]:
        """
        Transforms a static 3D landmark in base_link from ego pose at `from_stamp`
        to ego pose at `to_stamp` using SE(2) ego-motion compensation.
        """
        dt = to_stamp - from_stamp
        if abs(dt) <= 1e-4 or abs(dt) > 0.25:
            return point
        step = self._compute_ego_motion_step(min(from_stamp, to_stamp), max(from_stamp, to_stamp))
        if step is None:
            return point
        sign = 1.0 if dt >= 0.0 else -1.0
        abs_dt = abs(dt)
        dtheta = sign * (step.dyaw if step.dyaw is not None else step.wz * abs_dt)
        if step.dx is not None and step.dy is not None and dt >= 0.0:
            dx = step.dx
            dy = step.dy
        else:
            vx = sign * step.vx
            vy = sign * step.vy
            if abs(dtheta) < 1e-5:
                dx = vx * abs_dt
                dy = vy * abs_dt
            else:
                s = math.sin(dtheta) / dtheta
                c = (1.0 - math.cos(dtheta)) / dtheta
                dx = (vx * s - vy * c) * abs_dt
                dy = (vx * c + vy * s) * abs_dt
        tx = point[0] - dx
        ty = point[1] - dy
        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)
        return (
            float(cos_t * tx + sin_t * ty),
            float(-sin_t * tx + cos_t * ty),
            float(point[2]),
        )

    def _compute_ego_motion_step(
        self, prev_stamp: float, curr_stamp: float
    ) -> Optional[EgoMotionStep]:
        motion_buf = getattr(self, "motion_buffer", None)
        if not motion_buf:
            return None
        nearest_stamp, msg = min(
            motion_buf, key=lambda pair: abs(pair[0] - curr_stamp)
        )
        if abs(nearest_stamp - curr_stamp) > 0.5:
            return None

        vx, vy, wz = self._extract_twist_from_motion_msg(msg)
        dt = max(0.0, curr_stamp - prev_stamp) if prev_stamp > 0.0 else None
        if dt is None or dt <= 1e-6 or dt > 0.5:
            return EgoMotionStep(vx=vx, vy=vy, wz=wz, dt=dt)

        # Piecewise SE(2) integration over all buffered odometry samples in [prev_stamp, curr_stamp]
        in_window = [
            (s, self._extract_twist_from_motion_msg(m))
            for s, m in motion_buf
            if (prev_stamp - 0.02) <= s <= (curr_stamp + 0.02)
        ]
        if len(in_window) < 2:
            return EgoMotionStep(vx=vx, vy=vy, wz=wz, dt=dt)

        in_window.sort(key=lambda p: p[0])
        dx_acc = 0.0
        dy_acc = 0.0
        dyaw_acc = 0.0
        t_cursor = prev_stamp
        for i, (s_i, (vx_i, vy_i, wz_i)) in enumerate(in_window):
            t_next = min(curr_stamp, in_window[i + 1][0] if (i + 1) < len(in_window) else curr_stamp)
            sub_dt = max(0.0, t_next - t_cursor)
            if sub_dt <= 0.0:
                continue
            dth = wz_i * sub_dt
            c_yaw = math.cos(dyaw_acc + 0.5 * dth)
            s_yaw = math.sin(dyaw_acc + 0.5 * dth)
            dx_acc += (vx_i * c_yaw - vy_i * s_yaw) * sub_dt
            dy_acc += (vx_i * s_yaw + vy_i * c_yaw) * sub_dt
            dyaw_acc += dth
            t_cursor = t_next
            if t_cursor >= curr_stamp:
                break

        return EgoMotionStep(
            vx=vx,
            vy=vy,
            wz=wz,
            dt=dt,
            dx=dx_acc,
            dy=dy_acc,
            dyaw=dyaw_acc,
        )

    def _apply_tracker_step(
        self, observations: List[Observation2D], header: Any, stamp: float
    ) -> None:
        ego_motion = self._compute_ego_motion_step(
            self.tracker.last_step_stamp, stamp
        )
        tracks, lost_tracks, stats = self.tracker.step(
            observations, stamp, ego_motion=ego_motion
        )
        for key, value in stats.items():
            if key in self.counters:
                self.counters[key] += int(value)

        for track in lost_tracks:
            self._publish_lost_track(track)
        for track in tracks:
            self._publish_track(track, header)

        corridor: Optional[CorridorEstimate] = None
        if getattr(self, "enable_corridor_output", False):
            corridor = self.tracker.get_corridor_estimate(
                horizon_m=float(getattr(self, "corridor_horizon_m", 20.0)),
                step_m=float(getattr(self, "corridor_step_m", 1.0)),
                ego_motion=ego_motion,
            )
            self.last_corridor_estimate = corridor
            self._publish_corridor(corridor, header)

        if getattr(self, "enable_debug_viz", False):
            self._publish_debug_scene_markers(tracks, corridor, header, stamp)

    def _publish_corridor(
        self, corridor: Optional[CorridorEstimate], header: Any
    ) -> None:
        if corridor is None:
            return
        frame_id = getattr(self, "target_frame", "") or getattr(header, "frame_id", "base_link")
        stamp = getattr(header, "stamp", None)

        path_pub = getattr(self, "corridor_path_pub", None)
        if path_pub is not None:
            path_msg = Path()
            if stamp is not None:
                path_msg.header.stamp = stamp
            path_msg.header.frame_id = frame_id
            poses: List[PoseStamped] = []
            for wx, wy, heading, _kappa in corridor.centerline_waypoints:
                ps = PoseStamped()
                if stamp is not None:
                    ps.header.stamp = stamp
                ps.header.frame_id = frame_id
                ps.pose.position = Point(x=float(wx), y=float(wy), z=0.0)
                half_yaw = 0.5 * float(heading)
                ps.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=math.sin(half_yaw), w=math.cos(half_yaw)
                )
                poses.append(ps)
            path_msg.poses = poses
            path_pub.publish(path_msg)

        markers_pub = getattr(self, "corridor_markers_pub", None)
        if markers_pub is not None:
            marker_array = MarkerArray()
            specs = [
                (
                    "corridor_left",
                    0,
                    [(float(x), float(y), 0.05) for x, y in corridor.left_boundary_xy],
                    (0.12, 0.47, 0.96, 0.92),
                    0.08,
                ),
                (
                    "corridor_right",
                    1,
                    [(float(x), float(y), 0.05) for x, y in corridor.right_boundary_xy],
                    (0.98, 0.85, 0.10, 0.92),
                    0.08,
                ),
                (
                    "corridor_centerline",
                    2,
                    [(float(wx), float(wy), 0.08) for wx, wy, _h, _k in corridor.centerline_waypoints],
                    (0.15, 0.95, 0.35, 0.95),
                    0.10,
                ),
            ]
            for ns, mid, pts, rgba, width in specs:
                if not pts:
                    continue
                m = Marker()
                if stamp is not None:
                    m.header.stamp = stamp
                m.header.frame_id = frame_id
                m.ns = ns
                m.id = mid
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                m.scale.x = float(width)
                m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
                m.points = [Point(x=px, y=py, z=pz) for px, py, pz in pts]
                marker_array.markers.append(m)
            markers_pub.publish(marker_array)

        if "corridor_published" in self.counters:
            self.counters["corridor_published"] += 1

    def _publish_debug_scene_markers(
        self,
        tracks: List[TrackEstimate],
        corridor: Optional[CorridorEstimate],
        header: Any,
        stamp: float,
    ) -> None:
        frame_id = getattr(self, "target_frame", "") or getattr(header, "frame_id", "base_link")
        msg_stamp = getattr(header, "stamp", None)
        descriptors = build_scene_marker_descriptors(
            tracks=tracks,
            corridor=corridor,
            frame_id=frame_id,
            stamp_sec=stamp,
            confirmed_only=False,
        )
        self.last_debug_markers = descriptors
        markers_pub = getattr(self, "debug_markers_pub", None)
        if markers_pub is not None:
            type_map = {
                "CYLINDER": Marker.CYLINDER,
                "ELLIPSE": Marker.CYLINDER,
                "TEXT_VIEW_FACING": Marker.TEXT_VIEW_FACING,
                "LINE_STRIP": Marker.LINE_STRIP,
            }
            marker_array = MarkerArray()
            for desc in descriptors:
                m = Marker()
                if msg_stamp is not None:
                    m.header.stamp = msg_stamp
                m.header.frame_id = desc.frame_id or frame_id
                m.ns = desc.ns
                m.id = int(desc.marker_id)
                m.type = type_map.get(desc.marker_type, Marker.CYLINDER)
                m.action = Marker.ADD
                m.pose.position = Point(x=float(desc.x), y=float(desc.y), z=float(desc.z))
                half_yaw = 0.5 * float(desc.yaw_rad)
                m.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=math.sin(half_yaw), w=math.cos(half_yaw)
                )
                m.scale.x = float(desc.scale_x)
                m.scale.y = float(desc.scale_y)
                m.scale.z = float(desc.scale_z)
                m.color = ColorRGBA(
                    r=float(desc.rgba[0]),
                    g=float(desc.rgba[1]),
                    b=float(desc.rgba[2]),
                    a=float(desc.rgba[3]),
                )
                m.text = str(desc.text)
                if desc.points_xyz:
                    m.points = [
                        Point(x=float(px), y=float(py), z=float(pz))
                        for px, py, pz in desc.points_xyz
                    ]
                marker_array.markers.append(m)
            markers_pub.publish(marker_array)
        if "debug_markers_published" in self.counters:
            self.counters["debug_markers_published"] += 1

    def _publish_reprojection_debug(
        self,
        clusters_xyz: List[Tuple[float, float, float]],
        detections: List[Any],
        header: Any,
    ) -> None:
        annotations = build_reprojection_debug_annotations(
            clusters_xyz=clusters_xyz,
            detections=detections,
            intrinsics=self.projector.intrinsics,
            extrinsics=self.projector.extrinsics,
        )
        self.last_reprojection_annotations = annotations
        reproj_pub = getattr(self, "debug_reprojection_pub", None)
        if reproj_pub is not None:
            payload = [
                {
                    "cluster_index": int(a.cluster_index),
                    "u_proj": round(float(a.u_proj), 2),
                    "v_proj": round(float(a.v_proj), 2),
                    "depth_m": round(float(a.depth_m), 3),
                    "matched_detection_index": int(a.matched_detection_index),
                    "residual_px": round(float(a.residual_px), 2),
                    "class_label": str(a.class_label),
                }
                for a in annotations
            ]
            reproj_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "frame_id": getattr(header, "frame_id", ""),
                            "annotations": payload,
                        }
                    )
                )
            )

    def run_online_calibration(
        self, apply_extrinsics: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        samples = list(getattr(self, "calibration_window", []))
        if not samples:
            return None
        intrinsics = self.projector.intrinsics
        extrinsics = self.projector.extrinsics
        ext_res = calibrate_lidar_to_camera_extrinsics(samples, intrinsics, extrinsics)
        sync_res = calibrate_temporal_sync_offset(samples, intrinsics, extrinsics)
        if "online_calib_runs" in self.counters:
            self.counters["online_calib_runs"] += 1

        min_pairs = max(1, int(getattr(self, "online_calibration_min_pairs", 12)))
        should_apply = bool(
            getattr(self, "online_calibration_apply_extrinsics", False)
            if apply_extrinsics is None
            else apply_extrinsics
        )
        applied = False
        if (
            should_apply
            and ext_res.matched_pairs >= min_pairs
            and ext_res.final_rmse_px <= ext_res.initial_rmse_px + 1e-6
        ):
            self.projector.extrinsics = ext_res.calibrated_extrinsics
            profiles = getattr(self, "camera_profiles", None)
            if profiles and "primary" in profiles:
                profiles["primary"].projector.extrinsics = ext_res.calibrated_extrinsics
            applied = True
            if "online_calib_applied" in self.counters:
                self.counters["online_calib_applied"] += 1

        report = {
            "matched_pairs": int(ext_res.matched_pairs),
            "initial_rmse_px": float(ext_res.initial_rmse_px),
            "final_rmse_px": float(ext_res.final_rmse_px),
            "delta_roll_rad": float(ext_res.delta_roll_rad),
            "delta_pitch_rad": float(ext_res.delta_pitch_rad),
            "delta_yaw_rad": float(ext_res.delta_yaw_rad),
            "estimated_time_offset_sec": float(sync_res.estimated_time_offset_sec),
            "zero_offset_rmse_px": float(sync_res.zero_offset_rmse_px),
            "optimal_offset_rmse_px": float(sync_res.optimal_offset_rmse_px),
            "evaluated_frames": int(sync_res.evaluated_frames),
            "applied": applied,
        }
        self.last_online_calibration_report = report
        return report

    def _publish_track(self, track: TrackEstimate, header: Any) -> None:
        msg = ConeTrack()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.target_frame if self.target_frame else header.frame_id
        msg.track_id = track.track_id
        msg.position = Point(x=track.state[0], y=track.state[1], z=track.z)
        msg.velocity = Vector3(x=track.state[2], y=track.state[3], z=0.0)
        msg.covariance = [
            float(track.covariance[0][0]),
            float(track.covariance[0][1]),
            0.0,
            float(track.covariance[1][0]),
            float(track.covariance[1][1]),
            0.0,
            0.0,
            0.0,
            float(track.z_var),
        ]
        msg.class_probs = [float(v) for v in track.class_probs]
        msg.track_age = float(max(0.0, track.last_stamp - track.first_stamp))
        msg.track_status = int(track.status)
        msg.quality_score = float(track.quality)
        self.track_pub.publish(msg)

    def _publish_lost_track(self, track: TrackEstimate) -> None:
        now = self.get_clock().now().to_msg()
        msg = ConeTrack()
        msg.header.stamp = now
        msg.header.frame_id = self.target_frame
        msg.track_id = track.track_id
        msg.position = Point(x=track.state[0], y=track.state[1], z=track.z)
        msg.velocity = Vector3(x=track.state[2], y=track.state[3], z=0.0)
        msg.covariance = [
            float(track.covariance[0][0]),
            float(track.covariance[0][1]),
            0.0,
            float(track.covariance[1][0]),
            float(track.covariance[1][1]),
            0.0,
            0.0,
            0.0,
            float(track.z_var),
        ]
        msg.class_probs = [float(v) for v in track.class_probs]
        msg.track_age = float(max(0.0, track.last_stamp - track.first_stamp))
        msg.track_status = ConeTrack.TRACK_LOST
        msg.quality_score = float(track.quality)
        self.track_pub.publish(msg)

    def _aggregate_class_probs(
        self, camera_msg: Optional[CameraConeDetections]
    ) -> List[float]:
        if camera_msg is None or not camera_msg.detections:
            return [0.25, 0.25, 0.25, 0.25]

        acc = [0.0, 0.0, 0.0, 0.0]
        weight_sum = 0.0
        for det in camera_msg.detections:
            w = max(0.01, float(det.confidence))
            for i in range(4):
                acc[i] += float(det.class_probs[i]) * w
            weight_sum += w
        if weight_sum <= 1e-6:
            return [0.25, 0.25, 0.25, 0.25]
        probs = [max(0.0, v / weight_sum) for v in acc]
        norm = sum(probs)
        if norm <= 1e-6:
            return [0.25, 0.25, 0.25, 0.25]
        return [float(v / norm) for v in probs]

    def _semantic_weight(self, camera_msg: Optional[CameraConeDetections]) -> float:
        if camera_msg is None:
            return 0.6
        if not camera_msg.detections:
            return 0.7
        avg_conf = sum(float(det.confidence) for det in camera_msg.detections) / len(
            camera_msg.detections
        )
        return float(max(0.3, min(1.0, avg_conf)))

    def _process_camera_fallback_without_lidar(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        nearest_camera = self._nearest_camera(now_sec)
        if nearest_camera is None:
            self.counters["camera_fallback_no_camera"] += 1
            return
        cam_stamp, camera_msg = nearest_camera
        if (now_sec - cam_stamp) > self.camera_fallback_max_age_s:
            self.counters["camera_fallback_stale_camera"] += 1
            return
        if cam_stamp <= self.last_camera_fallback_stamp + 1e-9:
            self.counters["camera_fallback_duplicate_frame"] += 1
            return

        self._update_motion_alignment(cam_stamp)
        fallback_obs, fallback_tracker = self._build_camera_fallback_observations(
            camera_msg=camera_msg,
            stamp_sec=cam_stamp,
            source="camera_fallback_live",
        )
        if not fallback_obs:
            return
        self._publish_observation_messages(fallback_obs)
        self.last_obs_count = len(fallback_obs)
        self.last_camera_fallback_stamp = cam_stamp
        self.counters["camera_fallback_frames"] += 1
        self.counters["camera_fallback_observations"] += len(fallback_obs)

        if self.enable_tracks:
            self._apply_tracker_step(
                observations=fallback_tracker,
                header=camera_msg.header,
                stamp=cam_stamp,
            )

    def _build_camera_fallback_observations(
        self,
        camera_msg: CameraConeDetections,
        stamp_sec: float,
        source: str,
    ) -> Tuple[List[ConeObservation], List[Observation2D]]:
        observations: List[ConeObservation] = []
        tracker_observations: List[Observation2D] = []
        if not camera_msg.detections:
            return observations, tracker_observations

        frame_id = self.target_frame if self.target_frame else camera_msg.header.frame_id
        health = SENSOR_HEALTH_DEGRADED
        self.last_health = max(self.last_health, health)
        cam_stamp_s = (
            self._stamp_to_sec(camera_msg.header.stamp.sec, camera_msg.header.stamp.nanosec)
            if hasattr(camera_msg, "header") and getattr(camera_msg.header, "stamp", None) is not None
            else stamp_sec
        )
        stereo_frame = (
            self._nearest_depth_for_id(cam_stamp_s, camera_id="primary")
            if hasattr(self, "_nearest_depth_for_id")
            else None
        )

        projector = getattr(self, "projector", None)
        if projector is not None and hasattr(projector, "recover_unmatched_detections"):
            det_records = projector.detections_from_msg(camera_msg)
            recovered = projector.recover_unmatched_detections(
                detections=det_records,
                matched_detection_indices=[],
                existing_positions=[],
                cone_height_m=float(getattr(self, "camera_fallback_cone_height_m", 0.30)),
                min_confidence=float(getattr(self, "camera_fallback_min_confidence", 0.25)),
                min_bbox_height_px=float(getattr(self, "camera_fallback_min_bbox_height_px", 8.0)),
                min_depth_m=float(getattr(self, "camera_fallback_min_depth_m", 2.0)),
                max_depth_m=float(getattr(self, "camera_fallback_max_depth_m", 35.0)),
                dedup_radius_m=0.5,
                enable_ground_plane_depth_fusion=bool(
                    getattr(self, "enable_ground_plane_depth_fusion", True)
                ),
                ground_z_base_m=float(getattr(self, "ground_z_base_m", -0.22)),
                ground_slope_rad=float(getattr(self, "ground_slope_rad", 0.0)),
                stereo_depth=stereo_frame,
                enable_stereo_depth_fusion=bool(
                    getattr(self, "enable_stereo_depth_fusion", True)
                ),
            )
            img_q = float(getattr(camera_msg, "image_quality", 1.0))
            for rec in recovered:
                obs = ConeObservation()
                obs.header.stamp = camera_msg.header.stamp
                obs.header.frame_id = frame_id
                obs.position = Point(
                    x=float(rec.position[0]),
                    y=float(rec.position[1]),
                    z=float(rec.position[2]),
                )
                base_cov = (
                    list(rec.covariance)
                    if rec.used_stereo
                    else [
                        max(float(rec.covariance[0]), float(self.camera_fallback_cov_xy_m2)),
                        float(rec.covariance[1]),
                        0.0,
                        float(rec.covariance[3]),
                        max(float(rec.covariance[4]), float(self.camera_fallback_cov_xy_m2)),
                        0.0,
                        0.0,
                        0.0,
                        max(float(rec.covariance[8]), float(self.camera_fallback_cov_z_m2)),
                    ]
                )
                obs.covariance = self._scaled_covariance(
                    covariance=base_cov,
                    health=health,
                    projection_residual_mean_px=0.0,
                    camera_lidar_abs_delta_ms=self.last_camera_lidar_abs_delta_ms,
                    motion_age_ms=self.last_motion_age_ms,
                    motion_comp_error_m=self.last_motion_comp_error_m,
                )
                obs.class_probs = [float(v) for v in rec.class_probs]
                q_scale = max(float(self.camera_fallback_quality_scale), 0.75 if rec.used_stereo else 0.0)
                obs.quality_score = float(max(0.0, min(1.0, rec.confidence * img_q * q_scale)))
                obs.sensor_health = health
                obs.source = f"{source}_stereo" if rec.used_stereo else source
                observations.append(obs)
                if rec.used_stereo and hasattr(self, "counters") and "stereo_depth_fused_observations" in self.counters:
                    self.counters["stereo_depth_fused_observations"] += 1
                if getattr(self, "enable_tracks", True):
                    tracker_observations.append(self._to_tracker_observation(obs))
            return observations, tracker_observations

        for det in camera_msg.detections:
            confidence = float(max(0.0, min(1.0, det.confidence)))
            if confidence < self.camera_fallback_min_confidence:
                continue
            height_px = float(max(0.0, det.ymax - det.ymin))
            if height_px < self.camera_fallback_min_bbox_height_px:
                continue

            u = 0.5 * (float(det.xmin) + float(det.xmax))
            v = 0.5 * (float(det.ymin) + float(det.ymax))
            depth = self.camera_fallback_cone_height_m * self.camera_fy / max(1e-3, height_px)
            depth = float(
                max(self.camera_fallback_min_depth_m, min(self.camera_fallback_max_depth_m, depth))
            )
            point = self._camera_pixel_to_lidar_point(u=u, v=v, depth=depth)
            if point is None:
                continue

            obs = ConeObservation()
            obs.header.stamp = camera_msg.header.stamp
            obs.header.frame_id = frame_id
            obs.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            base_cov = [
                float(self.camera_fallback_cov_xy_m2),
                0.0,
                0.0,
                0.0,
                float(self.camera_fallback_cov_xy_m2),
                0.0,
                0.0,
                0.0,
                float(self.camera_fallback_cov_z_m2),
            ]
            obs.covariance = self._scaled_covariance(
                covariance=base_cov,
                health=health,
                projection_residual_mean_px=0.0,
                camera_lidar_abs_delta_ms=self.last_camera_lidar_abs_delta_ms,
                motion_age_ms=self.last_motion_age_ms,
                motion_comp_error_m=self.last_motion_comp_error_m,
            )
            obs.class_probs = [float(v) for v in det.class_probs]
            quality = float(
                max(
                    0.0,
                    min(
                        1.0,
                        confidence
                        * float(camera_msg.image_quality)
                        * self.camera_fallback_quality_scale,
                    ),
                )
            )
            obs.quality_score = quality
            obs.sensor_health = health
            obs.source = source
            observations.append(obs)
            if getattr(self, "enable_tracks", True):
                tracker_observations.append(self._to_tracker_observation(obs))

        return observations, tracker_observations

    def _camera_pixel_to_lidar_point(
        self, u: float, v: float, depth: float
    ) -> Optional[Tuple[float, float, float]]:
        fx = max(1e-6, float(self.camera_fx))
        fy = max(1e-6, float(self.camera_fy))
        x_cam = (float(u) - float(self.camera_cx)) * depth / fx
        y_cam = (float(v) - float(self.camera_cy)) * depth / fy
        z_cam = float(depth)

        r = self.projector.extrinsics.rotation
        t = self.projector.extrinsics.translation
        x0 = x_cam - float(t[0])
        y0 = y_cam - float(t[1])
        z0 = z_cam - float(t[2])
        # p_lidar = R^T * (p_cam - t)
        x_lidar = r[0][0] * x0 + r[1][0] * y0 + r[2][0] * z0
        y_lidar = r[0][1] * x0 + r[1][1] * y0 + r[2][1] * z0
        z_lidar = r[0][2] * x0 + r[1][2] * y0 + r[2][2] * z0
        if not (abs(x_lidar) < 1e6 and abs(y_lidar) < 1e6 and abs(z_lidar) < 1e6):
            return None
        return float(x_lidar), float(y_lidar), float(z_lidar)

    def _scaled_covariance(
        self,
        covariance: List[float],
        health: int,
        projection_residual_mean_px: float = 0.0,
        camera_lidar_abs_delta_ms: float = 0.0,
        motion_age_ms: float = 0.0,
        motion_comp_error_m: float = 0.0,
    ) -> List[float]:
        if len(covariance) != 9:
            covariance = [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.5]
        if all(abs(v) < 1e-12 for v in covariance):
            covariance = [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.5]

        scale = 1.0
        h_degrade = float(getattr(self, "health_degrade_scale", 1.5))
        if health == SENSOR_HEALTH_DEGRADED:
            scale = h_degrade
        elif health == SENSOR_HEALTH_CRITICAL:
            scale = h_degrade * 2.0

        proj_warn = float(getattr(self, "projection_residual_warn_px", 5.0))
        proj_crit = float(getattr(self, "projection_residual_critical_px", 12.0))
        if projection_residual_mean_px > proj_warn:
            if projection_residual_mean_px >= proj_crit:
                scale *= 2.0
            else:
                span = max(1e-6, proj_crit - proj_warn)
                ratio = (projection_residual_mean_px - proj_warn) / span
                scale *= 1.0 + 1.0 * max(0.0, min(1.0, ratio))

        sync_warn = float(getattr(self, "sync_offset_warn_ms", 40.0))
        sync_crit = float(getattr(self, "sync_offset_critical_ms", 90.0))
        if camera_lidar_abs_delta_ms > sync_warn:
            if camera_lidar_abs_delta_ms >= sync_crit:
                scale *= 1.75
            else:
                span = max(1e-6, sync_crit - sync_warn)
                ratio = (camera_lidar_abs_delta_ms - sync_warn) / span
                scale *= 1.0 + 0.75 * max(0.0, min(1.0, ratio))

        mot_warn = float(getattr(self, "motion_age_warn_ms", 80.0))
        mot_crit = float(getattr(self, "motion_age_critical_ms", 180.0))
        if motion_age_ms > mot_warn:
            if motion_age_ms >= mot_crit:
                scale *= 1.75
            else:
                span = max(1e-6, mot_crit - mot_warn)
                ratio = (motion_age_ms - mot_warn) / span
                scale *= 1.0 + 0.75 * max(0.0, min(1.0, ratio))

        comp_gain = float(getattr(self, "motion_comp_error_gain", 0.6))
        if motion_comp_error_m > 0.0:
            scale *= 1.0 + max(0.0, comp_gain) * motion_comp_error_m

        return [float(v * scale) for v in covariance]

    def _health_code(
        self, camera_msg: Optional[CameraConeDetections], lidar_msg: LidarConeClusters
    ) -> int:
        raw_lq = getattr(lidar_msg, "lidar_quality", None)
        lidar_quality = float(raw_lq) if raw_lq is not None and float(raw_lq) > 0.0 else 1.0
        q_crit = float(getattr(self, "quality_critical_threshold", 0.2))
        q_warn = float(getattr(self, "quality_warn_threshold", 0.4))
        if lidar_quality < q_crit:
            return SENSOR_HEALTH_CRITICAL
        if getattr(self, "use_tf_extrinsics", False) and not getattr(self, "last_tf_lookup_ok", True):
            return SENSOR_HEALTH_DEGRADED
        tf_age = float(getattr(self, "last_tf_age_ms", 0.0))
        if tf_age >= float(getattr(self, "tf_extrinsics_critical_age_ms", 120.0)):
            return SENSOR_HEALTH_CRITICAL
        if tf_age >= float(getattr(self, "tf_extrinsics_warn_age_ms", 40.0)):
            return SENSOR_HEALTH_DEGRADED
        sync_std = float(getattr(self, "sync_delta_std_ms", 0.0))
        if sync_std >= float(getattr(self, "sync_jitter_critical_ms", 25.0)):
            return SENSOR_HEALTH_CRITICAL
        if sync_std >= float(getattr(self, "sync_jitter_warn_ms", 10.0)):
            return SENSOR_HEALTH_DEGRADED
        mot_age = float(getattr(self, "last_motion_age_ms", 0.0))
        if mot_age >= float(getattr(self, "motion_age_critical_ms", 180.0)):
            return SENSOR_HEALTH_CRITICAL
        if mot_age >= float(getattr(self, "motion_age_warn_ms", 80.0)):
            return SENSOR_HEALTH_DEGRADED
        cam_lidar_delta = float(getattr(self, "last_camera_lidar_abs_delta_ms", 0.0))
        if cam_lidar_delta >= float(getattr(self, "sync_offset_critical_ms", 90.0)):
            return SENSOR_HEALTH_CRITICAL
        proj_res = float(getattr(self, "last_projection_residual_mean_px", 0.0))
        if proj_res >= float(getattr(self, "projection_residual_critical_px", 12.0)):
            return SENSOR_HEALTH_CRITICAL
        if camera_msg is None:
            return SENSOR_HEALTH_DEGRADED
        raw_cq = getattr(camera_msg, "image_quality", None)
        camera_quality = float(raw_cq) if raw_cq is not None and float(raw_cq) > 0.0 else 1.0
        if (
            camera_quality < q_warn
            or lidar_quality < q_warn
            or cam_lidar_delta >= float(getattr(self, "sync_offset_warn_ms", 40.0))
            or proj_res >= float(getattr(self, "projection_residual_warn_px", 5.0))
        ):
            return SENSOR_HEALTH_DEGRADED
        return SENSOR_HEALTH_OK

    def _on_camera_info(self, msg: Any, camera_id: str = "primary") -> None:
        if not getattr(self, "enable_camera_info_intrinsics", True):
            return
        k_mat = list(getattr(msg, "k", None) or getattr(msg, "K", []) or [])
        p_mat = list(getattr(msg, "p", None) or getattr(msg, "P", []) or [])
        d_vec = list(getattr(msg, "d", None) or getattr(msg, "D", []) or [])
        width = int(getattr(msg, "width", 0) or 0)
        height = int(getattr(msg, "height", 0) or 0)

        fx = fy = cx = cy = 0.0
        if len(k_mat) >= 9 and float(k_mat[0]) > 0.0 and float(k_mat[4]) > 0.0:
            fx = float(k_mat[0])
            cx = float(k_mat[2])
            fy = float(k_mat[4])
            cy = float(k_mat[5])
        elif len(p_mat) >= 12 and float(p_mat[0]) > 0.0 and float(p_mat[5]) > 0.0:
            fx = float(p_mat[0])
            cx = float(p_mat[2])
            fy = float(p_mat[5])
            cy = float(p_mat[6])
        if fx <= 0.0 or fy <= 0.0 or width <= 0 or height <= 0:
            return

        dist_model_raw = str(getattr(msg, "distortion_model", "plumb_bob") or "plumb_bob").strip().lower()
        if dist_model_raw in ("equidistant", "kannala_brandt", "fisheye"):
            dist_model = "equidistant"
            k1 = float(d_vec[0]) if len(d_vec) > 0 else 0.0
            k2 = float(d_vec[1]) if len(d_vec) > 1 else 0.0
            p1 = 0.0
            p2 = 0.0
            k3 = float(d_vec[2]) if len(d_vec) > 2 else 0.0
            k4 = float(d_vec[3]) if len(d_vec) > 3 else 0.0
        else:
            dist_model = "plumb_bob"
            k1 = float(d_vec[0]) if len(d_vec) > 0 else 0.0
            k2 = float(d_vec[1]) if len(d_vec) > 1 else 0.0
            p1 = float(d_vec[2]) if len(d_vec) > 2 else 0.0
            p2 = float(d_vec[3]) if len(d_vec) > 3 else 0.0
            k3 = float(d_vec[4]) if len(d_vec) > 4 else 0.0
            k4 = 0.0

        new_intr = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            k1=k1,
            k2=k2,
            p1=p1,
            p2=p2,
            k3=k3,
            k4=k4,
            distortion_model=dist_model,
        )
        profiles = getattr(self, "camera_profiles", None)
        if profiles and camera_id in profiles:
            profiles[camera_id].projector.intrinsics = new_intr
        if camera_id == "primary" and getattr(self, "projector", None) is not None:
            self.projector.intrinsics = new_intr

        counters = getattr(self, "counters", None)
        if isinstance(counters, dict):
            counters["camera_info_updates"] = int(counters.get("camera_info_updates", 0)) + 1
        last_map = getattr(self, "last_camera_info_cameras", None)
        if isinstance(last_map, dict):
            last_map[camera_id] = (fx, fy, cx, cy, width, height)

    def _refresh_projector_extrinsics(
        self,
        lidar_msg: LidarConeClusters,
        camera_msg: Optional[CameraConeDetections],
        camera_msgs_by_cam: Optional[Dict[str, CameraConeDetections]] = None,
    ) -> None:
        if not self.use_tf_extrinsics or self.tf_buffer is None:
            self.last_extrinsics_source = "params"
            self.last_tf_lookup_ok = True
            self.last_tf_lookup_error = ""
            self.last_tf_age_ms = 0.0
            return

        lidar_frame = self.tf_extrinsics_lidar_frame.strip()
        if not lidar_frame:
            lidar_frame = str(lidar_msg.header.frame_id).strip()
        camera_frame = self.tf_extrinsics_camera_frame.strip()
        if not camera_frame and camera_msg is not None:
            camera_frame = str(camera_msg.header.frame_id).strip()
        if not lidar_frame or not camera_frame:
            self.counters["tf_extrinsics_missing_frame"] += 1
            self.last_tf_lookup_ok = False
            self.last_tf_lookup_error = "missing lidar/camera frame for tf lookup"
            self.last_extrinsics_source = "params"
            return

        source_time = Time.from_msg(lidar_msg.header.stamp)
        timeout = Duration(seconds=max(0.0, self.tf_lookup_timeout_ms * 1e-3))
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                camera_frame, lidar_frame, source_time, timeout=timeout
            )
        except TransformException as exc:
            self.counters["tf_lookup_fail"] += 1
            self.last_tf_lookup_ok = False
            self.last_tf_lookup_error = str(exc)
            self.last_extrinsics_source = "params"
            return

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        self.projector.extrinsics = LidarToCameraExtrinsics(
            rotation=self._quat_to_rotation_matrix(
                float(q.x), float(q.y), float(q.z), float(q.w)
            ),
            translation=(float(t.x), float(t.y), float(t.z)),
        )
        self.counters["tf_lookup_success"] += 1
        self.last_tf_lookup_ok = True
        self.last_tf_lookup_error = ""
        self.last_extrinsics_source = "tf"
        self.last_tf_lidar_frame = lidar_frame
        self.last_tf_camera_frame = camera_frame

        tf_stamp = self._stamp_to_sec(
            tf_msg.header.stamp.sec, tf_msg.header.stamp.nanosec
        )
        lidar_stamp = self._stamp_to_sec(
            lidar_msg.header.stamp.sec, lidar_msg.header.stamp.nanosec
        )
        if tf_stamp <= 0.0:
            self.last_tf_age_ms = 0.0
        else:
            self.last_tf_age_ms = max(0.0, (lidar_stamp - tf_stamp) * 1000.0)
        if self.last_tf_age_ms >= self.tf_extrinsics_warn_age_ms:
            self.counters["tf_extrinsics_stale"] += 1

        # Refresh any secondary / N-camera profiles that opt in to dynamic TF extrinsics
        profiles = getattr(self, "camera_profiles", None) or {}
        for cam_id, prof in profiles.items():
            if cam_id == "primary" or not getattr(prof, "use_tf_extrinsics", False):
                continue
            aux_cam_frame = str(getattr(prof, "frame_id", "")).strip()
            if not aux_cam_frame and camera_msgs_by_cam and cam_id in camera_msgs_by_cam:
                aux_cam_frame = str(getattr(camera_msgs_by_cam[cam_id].header, "frame_id", "")).strip()
            if not aux_cam_frame:
                continue
            try:
                aux_tf = self.tf_buffer.lookup_transform(
                    aux_cam_frame, lidar_frame, source_time, timeout=timeout
                )
            except TransformException:
                self.counters["tf_lookup_fail"] += 1
                continue
            at = aux_tf.transform.translation
            aq = aux_tf.transform.rotation
            prof.projector.extrinsics = LidarToCameraExtrinsics(
                rotation=self._quat_to_rotation_matrix(
                    float(aq.x), float(aq.y), float(aq.z), float(aq.w)
                ),
                translation=(float(at.x), float(at.y), float(at.z)),
            )
            self.counters["tf_lookup_success"] += 1

    @staticmethod
    def _quat_to_rotation_matrix(
        x: float, y: float, z: float, w: float
    ) -> List[List[float]]:
        n = x * x + y * y + z * z + w * w
        if n <= 1e-12:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        s = 2.0 / n
        xx = x * x * s
        yy = y * y * s
        zz = z * z * s
        xy = x * y * s
        xz = x * z * s
        yz = y * z * s
        wx = w * x * s
        wy = w * y * s
        wz = w * z * s
        return [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]

    def _projection_residual_penalty(self, residual_mean_px: float) -> float:
        if residual_mean_px <= self.projection_residual_warn_px:
            return 1.0
        if residual_mean_px >= self.projection_residual_critical_px:
            return 0.5
        span = max(1e-6, self.projection_residual_critical_px - self.projection_residual_warn_px)
        ratio = (residual_mean_px - self.projection_residual_warn_px) / span
        return float(max(0.5, min(1.0, 1.0 - 0.5 * ratio)))

    def _update_motion_alignment(self, reference_stamp: float) -> None:
        if not self.motion_buffer:
            self.last_motion_age_ms = float("inf")
            self.last_motion_comp_error_m = 0.0
            self.counters["motion_missing_for_covariance"] += 1
            return
        nearest_stamp, msg = min(
            self.motion_buffer, key=lambda pair: abs(pair[0] - reference_stamp)
        )
        age_ms = abs(nearest_stamp - reference_stamp) * 1000.0
        self.last_motion_age_ms = float(age_ms)
        self.max_motion_age_ms = max(self.max_motion_age_ms, self.last_motion_age_ms)

        vx, vy, _wz = self._extract_twist_from_motion_msg(msg)
        tw_outer = getattr(msg, "twist", None)
        tw = getattr(tw_outer, "twist", None) or tw_outer
        lin = getattr(tw, "linear", None)
        vz = float(getattr(lin, "z", 0.0)) if lin is not None else 0.0
        speed = (vx * vx + vy * vy + vz * vz) ** 0.5
        self.last_motion_comp_error_m = float(speed * age_ms * 1e-3)
        self.max_motion_comp_error_m = max(
            self.max_motion_comp_error_m, self.last_motion_comp_error_m
        )

    def _update_camera_lidar_sync(self, camera_stamp: float, lidar_stamp: float) -> None:
        delta_ms = (camera_stamp - lidar_stamp) * 1000.0
        abs_delta_ms = abs(delta_ms)
        self.last_camera_lidar_delta_ms = float(delta_ms)
        self.last_camera_lidar_abs_delta_ms = float(abs_delta_ms)
        self.max_camera_lidar_abs_delta_ms = max(
            self.max_camera_lidar_abs_delta_ms, self.last_camera_lidar_abs_delta_ms
        )
        self.camera_lidar_delta_samples += 1
        self.sync_delta_window_ms.append(self.last_camera_lidar_delta_ms)
        self._update_sync_statistics()

    def _update_sync_statistics(self) -> None:
        if not self.sync_delta_window_ms:
            self.sync_delta_mean_ms = 0.0
            self.sync_delta_std_ms = 0.0
            self.sync_delta_abs_p95_ms = 0.0
            return
        values = list(self.sync_delta_window_ms)
        n = float(len(values))
        mean = sum(values) / n
        var = sum((v - mean) * (v - mean) for v in values) / n
        abs_values = sorted(abs(v) for v in values)
        idx = max(0, min(len(abs_values) - 1, int(math.ceil(0.95 * len(abs_values))) - 1))
        self.sync_delta_mean_ms = float(mean)
        self.sync_delta_std_ms = float(math.sqrt(max(0.0, var)))
        self.sync_delta_abs_p95_ms = float(abs_values[idx])

    def _update_latency_and_blind_distance(self, lidar_stamp: float) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.last_latency_ms = max(0.0, (now_sec - lidar_stamp) * 1000.0)
        self.max_latency_ms = max(self.max_latency_ms, self.last_latency_ms)
        self.last_blind_distance_current_m = (
            self.last_ego_speed_mps * self.last_latency_ms * 1e-3
        )
        self.last_blind_distance_target_m = (
            self.blind_distance_target_speed_mps * self.last_latency_ms * 1e-3
        )
        self.max_blind_distance_target_m = max(
            self.max_blind_distance_target_m, self.last_blind_distance_target_m
        )

    @staticmethod
    def _elapsed_ms(start_ns: int) -> float:
        return max(0.0, (time.perf_counter_ns() - start_ns) * 1e-6)

    def _record_stage_timing(self, stage: str, elapsed_ms: float) -> None:
        if stage not in self.last_stage_ms:
            return
        elapsed = float(max(0.0, elapsed_ms))
        self.last_stage_ms[stage] = elapsed
        self.max_stage_ms[stage] = max(self.max_stage_ms.get(stage, 0.0), elapsed)

    def _publish_diagnostics(self) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = "fusion_core/runtime"
        status.hardware_id = "fusion_core"
        status.level = DiagnosticStatus.OK
        status.message = "ok"

        if self.last_health == SENSOR_HEALTH_CRITICAL:
            status.level = DiagnosticStatus.ERROR
            status.message = "critical sensor health"
        elif (
            self.last_health == SENSOR_HEALTH_DEGRADED
            or self.counters["camera_overflow"] > 0
            or self.counters["lidar_overflow"] > 0
            or self.counters["oosm_dropped"] > 0
            or self.counters["replay_budget_exceeded"] > 0
            or self.last_camera_lidar_abs_delta_ms >= self.sync_offset_warn_ms
            or self.sync_delta_std_ms >= self.sync_jitter_warn_ms
            or self.last_projection_residual_mean_px >= self.projection_residual_warn_px
            or self.last_motion_age_ms >= self.motion_age_warn_ms
            or getattr(self, "last_extrinsics_sanity", "ok")
            not in ("ok", "no_clusters")
            or (
                self.use_tf_extrinsics
                and (
                    not self.last_tf_lookup_ok
                    or self.last_tf_age_ms >= self.tf_extrinsics_warn_age_ms
                )
            )
        ):
            status.level = DiagnosticStatus.WARN
            if getattr(self, "last_extrinsics_sanity", "ok") not in ("ok", "no_clusters"):
                status.message = f"extrinsics sanity: {self.last_extrinsics_sanity}"
            else:
                status.message = "degraded or dropped data detected"

        projection_residual_mean_total_px = (
            self.projection_residual_sum_px_total
            / float(self.projection_residual_samples_total)
            if self.projection_residual_samples_total > 0
            else 0.0
        )
        values = [
            ("camera_buffer_size", str(len(self.camera_buffer))),
            ("lidar_buffer_size", str(len(self.lidar_buffer))),
            ("motion_buffer_size", str(len(self.motion_buffer))),
            ("tracks_active", str(len(self.tracker.tracks))),
            ("frame_history_size", str(len(self.frame_history))),
            ("last_observation_count", str(self.last_obs_count)),
            ("last_latency_ms", f"{self.last_latency_ms:.2f}"),
            ("max_latency_ms", f"{self.max_latency_ms:.2f}"),
            ("last_ego_speed_mps", f"{self.last_ego_speed_mps:.3f}"),
            ("blind_distance_target_speed_mps", f"{self.blind_distance_target_speed_mps:.3f}"),
            ("last_blind_distance_current_m", f"{self.last_blind_distance_current_m:.3f}"),
            ("last_blind_distance_target_m", f"{self.last_blind_distance_target_m:.3f}"),
            ("max_blind_distance_target_m", f"{self.max_blind_distance_target_m:.3f}"),
            ("last_camera_lidar_delta_ms", f"{self.last_camera_lidar_delta_ms:.3f}"),
            ("last_camera_lidar_abs_delta_ms", f"{self.last_camera_lidar_abs_delta_ms:.3f}"),
            ("max_camera_lidar_abs_delta_ms", f"{self.max_camera_lidar_abs_delta_ms:.3f}"),
            ("camera_lidar_delta_samples", str(self.camera_lidar_delta_samples)),
            ("sync_delta_mean_ms", f"{self.sync_delta_mean_ms:.3f}"),
            ("sync_delta_std_ms", f"{self.sync_delta_std_ms:.3f}"),
            ("sync_delta_abs_p95_ms", f"{self.sync_delta_abs_p95_ms:.3f}"),
            ("last_motion_age_ms", f"{self.last_motion_age_ms:.3f}"),
            ("max_motion_age_ms", f"{self.max_motion_age_ms:.3f}"),
            ("last_motion_comp_error_m", f"{self.last_motion_comp_error_m:.3f}"),
            ("max_motion_comp_error_m", f"{self.max_motion_comp_error_m:.3f}"),
            ("last_projection_residual_mean_px", f"{self.last_projection_residual_mean_px:.3f}"),
            ("projection_residual_max_px", f"{self.projection_residual_max_px:.3f}"),
            ("projection_residual_samples_total", str(self.projection_residual_samples_total)),
            ("projection_residual_mean_total_px", f"{projection_residual_mean_total_px:.3f}"),
            ("use_tf_extrinsics", str(self.use_tf_extrinsics).lower()),
            ("last_extrinsics_source", self.last_extrinsics_source),
            ("last_extrinsics_sanity", str(getattr(self, "last_extrinsics_sanity", "ok"))),
            ("last_tf_lookup_ok", str(self.last_tf_lookup_ok).lower()),
            ("last_tf_age_ms", f"{self.last_tf_age_ms:.3f}"),
            ("last_tf_lidar_frame", self.last_tf_lidar_frame),
            ("last_tf_camera_frame", self.last_tf_camera_frame),
            ("last_tf_lookup_error", self.last_tf_lookup_error),
            ("stage_ms_camera_lookup", f"{self.last_stage_ms['camera_lookup']:.3f}"),
            ("stage_ms_extrinsics_lookup", f"{self.last_stage_ms['extrinsics_lookup']:.3f}"),
            ("stage_ms_build_observations", f"{self.last_stage_ms['build_observations']:.3f}"),
            ("stage_ms_publish_observations", f"{self.last_stage_ms['publish_observations']:.3f}"),
            ("stage_ms_tracker_step", f"{self.last_stage_ms['tracker_step']:.3f}"),
            ("stage_ms_replay_total", f"{self.last_stage_ms['replay_total']:.3f}"),
            ("stage_ms_cycle_total", f"{self.last_stage_ms['cycle_total']:.3f}"),
            ("stage_ms_max_camera_lookup", f"{self.max_stage_ms['camera_lookup']:.3f}"),
            ("stage_ms_max_extrinsics_lookup", f"{self.max_stage_ms['extrinsics_lookup']:.3f}"),
            ("stage_ms_max_build_observations", f"{self.max_stage_ms['build_observations']:.3f}"),
            ("stage_ms_max_publish_observations", f"{self.max_stage_ms['publish_observations']:.3f}"),
            ("stage_ms_max_tracker_step", f"{self.max_stage_ms['tracker_step']:.3f}"),
            ("stage_ms_max_replay_total", f"{self.max_stage_ms['replay_total']:.3f}"),
            ("stage_ms_max_cycle_total", f"{self.max_stage_ms['cycle_total']:.3f}"),
            ("last_health", str(self.last_health)),
            ("registered_cameras", str(len(getattr(self, "camera_profiles", {"primary": None})))),
            ("registered_lidars", str(len(getattr(self, "lidar_profiles", {"primary": None})))),
            ("enable_corridor_output", str(getattr(self, "enable_corridor_output", False)).lower()),
            ("enable_debug_viz", str(getattr(self, "enable_debug_viz", False)).lower()),
            ("enable_online_calibration", str(getattr(self, "enable_online_calibration", False)).lower()),
            ("online_calib_window_size", str(len(getattr(self, "calibration_window", ())))),
            (
                "replay_anchor_stamp",
                "none" if self.replay_anchor_stamp is None else f"{self.replay_anchor_stamp:.6f}",
            ),
        ]
        for key, value in self.counters.items():
            values.append((key, str(value)))
        status.values = [KeyValue(key=k, value=v) for k, v in values]

        msg.status = [status]
        self.diag_pub.publish(msg)

    @staticmethod
    def _stamp_to_sec(sec: int, nanosec: int) -> float:
        return float(sec) + float(nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
