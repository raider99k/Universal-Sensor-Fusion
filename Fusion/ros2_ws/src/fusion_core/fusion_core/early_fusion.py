from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:
    from .cpp_bridge import (
        CameraIntrinsicsC,
        ClusterInputC,
        ClusterSemanticC,
        Detection2DC,
        ExtrinsicsC,
        get_cpp_kernels,
    )
except ImportError:  # pragma: no cover - direct module import fallback
    from fusion_core.cpp_bridge import (  # type: ignore[no-redef]
        CameraIntrinsicsC,
        ClusterInputC,
        ClusterSemanticC,
        Detection2DC,
        ExtrinsicsC,
        get_cpp_kernels,
    )

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:
    np = None  # type: ignore
    linear_sum_assignment = None  # type: ignore
    _HAS_SCIPY = False

if TYPE_CHECKING:
    from fusion_interfaces.msg import CameraConeDetections


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_probs(probabilities: Sequence[float]) -> List[float]:
    probs = [max(0.0, float(v)) for v in probabilities[:4]]
    while len(probs) < 4:
        probs.append(0.0)
    total = sum(probs)
    if total <= 1e-9:
        return [0.25, 0.25, 0.25, 0.25]
    return [v / total for v in probs]


def _rpy_to_rotation_matrix(
    roll: float, pitch: float, yaw: float
) -> List[List[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]

    return _mat_mul_3x3(_mat_mul_3x3(rz, ry), rx)


def _mat_mul_3x3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [
            a[0][0] * b[0][j] + a[0][1] * b[1][j] + a[0][2] * b[2][j]
            for j in range(3)
        ],
        [
            a[1][0] * b[0][j] + a[1][1] * b[1][j] + a[1][2] * b[2][j]
            for j in range(3)
        ],
        [
            a[2][0] * b[0][j] + a[2][1] * b[1][j] + a[2][2] * b[2][j]
            for j in range(3)
        ],
    ]


def _mat_vec_mul_3x3(a: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def has_distortion(self) -> bool:
        return (
            abs(self.k1) > 1e-12
            or abs(self.k2) > 1e-12
            or abs(self.p1) > 1e-12
            or abs(self.p2) > 1e-12
            or abs(self.k3) > 1e-12
        )

    def distort_normalized(self, xn: float, yn: float) -> Tuple[float, float]:
        if not self.has_distortion():
            return xn, yn
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        xy2 = 2.0 * xn * yn
        xd = xn * radial + self.p1 * xy2 + self.p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + self.p2 * xy2
        return float(xd), float(yd)

    def undistort_normalized(self, xd: float, yd: float, iterations: int = 5) -> Tuple[float, float]:
        if not self.has_distortion():
            return xd, yd
        xn, yn = xd, yd
        for _ in range(max(1, iterations)):
            r2 = xn * xn + yn * yn
            r4 = r2 * r2
            r6 = r4 * r2
            radial = max(1e-4, 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6)
            xy2 = 2.0 * xn * yn
            dx_tang = self.p1 * xy2 + self.p2 * (r2 + 2.0 * xn * xn)
            dy_tang = self.p1 * (r2 + 2.0 * yn * yn) + self.p2 * xy2
            xn = (xd - dx_tang) / radial
            yn = (yd - dy_tang) / radial
        return float(xn), float(yn)


@dataclass
class LidarToCameraExtrinsics:
    rotation: List[List[float]]
    translation: Tuple[float, float, float]

    @staticmethod
    def from_rpy_translation(
        tx: float,
        ty: float,
        tz: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> "LidarToCameraExtrinsics":
        return LidarToCameraExtrinsics(
            rotation=_rpy_to_rotation_matrix(roll, pitch, yaw),
            translation=(tx, ty, tz),
        )


@dataclass
class DetectionRecord:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_probs: List[float]
    confidence: float
    is_big_orange: bool = False
    polygon_uv: Optional[List[Tuple[float, float]]] = None


@dataclass
class StereoDepthFrame:
    """Registered stereo depth or disparity image frame (e.g. Stereolabs ZED X / ZED 2)."""

    stamp: float
    width: int
    height: int
    encoding: str  # "32FC1", "16UC1", or "32FC1_disparity"
    data: bytes
    row_step_bytes: int
    depth_scale: float = 1.0
    baseline_m: float = 0.12
    disparity_sigma_px: float = 0.5
    min_depth_m: float = 0.5
    max_depth_m: float = 45.0


@dataclass
class ProjectionDiagnostics:
    projected_samples: int
    matched_samples: int
    matched_weight_sum: float
    best_confidence: float
    residual_samples: int
    residual_sum_px: float
    residual_max_px: float
    matched_detection_idx: Optional[int] = None


@dataclass
class ClusterSemanticEstimate:
    class_probs: List[float]
    semantic_weight: float
    diagnostics: ProjectionDiagnostics


@dataclass
class FrustumRecoveredCone:
    position: Tuple[float, float, float]
    covariance: List[float]  # 3x3 row-major
    class_probs: List[float]
    confidence: float
    depth_m: float
    detection_idx: int
    used_stereo: bool = False


def project_lidar_point_to_image(
    point_lidar: Tuple[float, float, float],
    intrinsics: CameraIntrinsics,
    extrinsics: LidarToCameraExtrinsics,
    min_depth_m: float,
    max_depth_m: float,
) -> Optional[Tuple[float, float, float]]:
    x_cam, y_cam, z_cam = _mat_vec_mul_3x3(extrinsics.rotation, point_lidar)
    x_cam += extrinsics.translation[0]
    y_cam += extrinsics.translation[1]
    z_cam += extrinsics.translation[2]

    if z_cam <= min_depth_m or z_cam >= max_depth_m:
        return None

    xn = x_cam / z_cam
    yn = y_cam / z_cam
    xd, yd = intrinsics.distort_normalized(xn, yn)
    u = intrinsics.fx * xd + intrinsics.cx
    v = intrinsics.fy * yd + intrinsics.cy

    if u < 0.0 or u >= float(intrinsics.width):
        return None
    if v < 0.0 or v >= float(intrinsics.height):
        return None
    return float(u), float(v), float(z_cam)


def _point_in_polygon(poly: Sequence[Tuple[float, float]], u: float, v: float) -> bool:
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > v) != (yj > v)) and (
            u < (xj - xi) * (v - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _min_dist_to_polygon_edges(poly: Sequence[Tuple[float, float]], u: float, v: float) -> float:
    n = len(poly)
    best_d2 = 1e18
    j = n - 1
    for i in range(n):
        x1, y1 = poly[j]
        x2, y2 = poly[i]
        dx = x2 - x1
        dy = y2 - y1
        len2 = dx * dx + dy * dy
        t = _clamp(((u - x1) * dx + (v - y1) * dy) / len2, 0.0, 1.0) if len2 > 1e-12 else 0.0
        px = x1 + t * dx
        py = y1 + t * dy
        d2 = (u - px) * (u - px) + (v - py) * (v - py)
        if d2 < best_d2:
            best_d2 = d2
        j = i
    return math.sqrt(best_d2)


def sample_stereo_depth_for_detection(
    det: DetectionRecord,
    stereo_depth: StereoDepthFrame,
    intrinsics: CameraIntrinsics,
) -> Optional[Tuple[float, float]]:
    """
    Sample ZED/stereo depth or disparity inside the foreground cone silhouette.
    Uses a 7x7 tapered interior grid and a 35th-percentile trimmed core estimator
    to reject background asphalt/sky behind the cone's bounding box corners.
    Returns (depth_m, sigma_depth_m) or None if insufficient valid samples.
    """
    import struct as _struct

    if not stereo_depth.data or stereo_depth.width <= 0 or stereo_depth.height <= 0:
        return None
    enc = str(stereo_depth.encoding).strip().lower()
    if enc in ("32fc1", "32fc1_meters"):
        mode = 1
    elif enc in ("16uc1", "16uc1_mm", "16uc1_millimeters"):
        mode = 2
    elif enc in ("32fc1_disparity", "disparity"):
        mode = 3
    else:
        return None

    w = max(1.0, float(det.xmax - det.xmin))
    h = max(1.0, float(det.ymax - det.ymin))
    cx = 0.5 * (float(det.xmin) + float(det.xmax))
    v0 = max(0.0, float(det.ymin) + 0.25 * h)
    v1 = min(float(max(1, intrinsics.height) - 1), float(det.ymin) + 0.85 * h)
    if v1 <= v0:
        return None

    fx = max(1.0, float(intrinsics.fx))
    baseline = float(stereo_depth.baseline_m) if stereo_depth.baseline_m > 1e-4 else 0.12
    disp_sigma = float(stereo_depth.disparity_sigma_px) if stereo_depth.disparity_sigma_px > 1e-4 else 0.5
    min_d = max(0.1, float(stereo_depth.min_depth_m))
    max_d = max(min_d + 0.1, float(stereo_depth.max_depth_m))
    scale_x = float(stereo_depth.width) / float(max(1, intrinsics.width))
    scale_y = float(stereo_depth.height) / float(max(1, intrinsics.height))
    depth_scale = float(stereo_depth.depth_scale) if stereo_depth.depth_scale > 0.0 else (0.001 if mode == 2 else 1.0)
    row_step = max(1, int(stereo_depth.row_step_bytes))
    buf = stereo_depth.data
    poly = det.polygon_uv if det.polygon_uv and len(det.polygon_uv) >= 3 else None

    samples: List[float] = []
    k_grid = 7
    for gy in range(k_grid):
        fy_t = (gy + 0.5) / float(k_grid)
        v_img = v0 + fy_t * (v1 - v0)
        py = max(0, min(stereo_depth.height - 1, int(round(v_img * scale_y))))
        t_cone = _clamp((v_img - float(det.ymin)) / h, 0.0, 1.0)
        half_span = max(1.0, 0.30 * w * (0.20 + 0.80 * t_cone))
        u_lo = max(0.0, cx - half_span)
        u_hi = min(float(max(1, intrinsics.width) - 1), cx + half_span)

        for gx in range(k_grid):
            fx_t = (gx + 0.5) / float(k_grid)
            u_img = u_lo + fx_t * (u_hi - u_lo)
            if poly is not None and not _point_in_polygon(poly, u_img, v_img):
                continue
            px = max(0, min(stereo_depth.width - 1, int(round(u_img * scale_x))))
            z_val = float("nan")
            if mode == 1:
                off = py * row_step + px * 4
                if off + 4 <= len(buf):
                    z_val = _struct.unpack_from("<f", buf, off)[0] * depth_scale
            elif mode == 2:
                off = py * row_step + px * 2
                if off + 2 <= len(buf):
                    raw_u = _struct.unpack_from("<H", buf, off)[0]
                    if raw_u > 0:
                        z_val = float(raw_u) * depth_scale
            elif mode == 3:
                off = py * row_step + px * 4
                if off + 4 <= len(buf):
                    disp = _struct.unpack_from("<f", buf, off)[0]
                    if math.isfinite(disp) and disp > 0.25:
                        z_val = (fx * baseline) / float(disp)
            if math.isfinite(z_val) and min_d <= z_val <= max_d:
                samples.append(float(z_val))

    if len(samples) < 3:
        return None
    samples.sort()
    n = len(samples)
    idx_p20 = int(0.20 * (n - 1))
    idx_p55 = int(0.55 * (n - 1))
    core = samples[idx_p20 : idx_p55 + 1]
    z_est = sum(core) / float(len(core))
    iqr = max(0.0, samples[idx_p55] - samples[idx_p20])
    sigma_triang = (z_est * z_est * disp_sigma) / max(1.0, fx * baseline)
    sigma_total = max(0.05, sigma_triang + 0.02 * z_est + 0.50 * iqr)
    return float(z_est), float(sigma_total)


class EarlyFusionProjector:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: LidarToCameraExtrinsics,
        min_depth_m: float = 0.1,
        max_depth_m: float = 80.0,
        sample_radius_scale: float = 1.5,
        min_semantic_weight: float = 0.35,
        no_match_semantic_weight: float = 0.6,
        soft_likelihood_enabled: bool = True,
        soft_likelihood_sigma_scale: float = 0.35,
        soft_likelihood_min_value: float = 0.02,
        soft_likelihood_outside_ratio: float = 1.6,
        aero_pitch_rad_per_mps2: float = 0.0,
        aero_tz_m_per_mps2: float = 0.0,
        enable_frustum_hull_sampling: bool = False,
        enable_triangular_cone_mask: bool = False,
        cone_mask_apex_width_ratio: float = 0.18,
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.base_extrinsics = LidarToCameraExtrinsics(
            rotation=[ list(row) for row in extrinsics.rotation ],
            translation=tuple(extrinsics.translation),
        )
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.sample_radius_scale = float(sample_radius_scale)
        self.min_semantic_weight = float(min_semantic_weight)
        self.no_match_semantic_weight = float(no_match_semantic_weight)
        self.soft_likelihood_enabled = bool(soft_likelihood_enabled)
        self.soft_likelihood_sigma_scale = float(soft_likelihood_sigma_scale)
        self.soft_likelihood_min_value = float(soft_likelihood_min_value)
        self.soft_likelihood_outside_ratio = float(soft_likelihood_outside_ratio)
        self.aero_pitch_rad_per_mps2 = float(aero_pitch_rad_per_mps2)
        self.aero_tz_m_per_mps2 = float(aero_tz_m_per_mps2)
        self.enable_frustum_hull_sampling = bool(enable_frustum_hull_sampling)
        self.enable_triangular_cone_mask = bool(enable_triangular_cone_mask)
        self.cone_mask_apex_width_ratio = float(_clamp(cone_mask_apex_width_ratio, 0.0, 1.0))

    def apply_aero_compensation(
        self,
        speed_mps: float,
        chassis_pitch_rad: float = 0.0,
        chassis_roll_rad: float = 0.0,
    ) -> None:
        """
        Dynamic aero-deflection (v^2 downforce) and live IMU/Odometry chassis attitude
        (braking nose-dive pitch + high-G cornering roll) compensation.
        """
        if (
            abs(self.aero_pitch_rad_per_mps2) <= 1e-12
            and abs(self.aero_tz_m_per_mps2) <= 1e-12
            and abs(chassis_pitch_rad) <= 1e-12
            and abs(chassis_roll_rad) <= 1e-12
        ):
            return
        v2 = max(0.0, float(speed_mps)) ** 2
        dpitch = self.aero_pitch_rad_per_mps2 * v2 + float(chassis_pitch_rad)
        droll = float(chassis_roll_rad)
        dtz = self.aero_tz_m_per_mps2 * v2
        cp, sp = math.cos(dpitch), math.sin(dpitch)
        cr, sr = math.cos(droll), math.sin(droll)
        # In OpenCV optical frame (x right, y down, z forward):
        # Pitch rotates around optical X axis; Roll rotates around optical Z axis.
        r_pitch = [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ]
        r_roll = [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ]
        r_dyn = _mat_mul_3x3(r_roll, r_pitch)
        self.extrinsics = LidarToCameraExtrinsics(
            rotation=_mat_mul_3x3(r_dyn, self.base_extrinsics.rotation),
            translation=(
                self.base_extrinsics.translation[0],
                self.base_extrinsics.translation[1] + dtz,
                self.base_extrinsics.translation[2],
            ),
        )

    def check_extrinsics_sanity(
        self, centroids_lidar: Sequence[Tuple[float, float, float]]
    ) -> str:
        """
        Verifies whether forward-facing LiDAR points project in front of the camera (z_cam > 0).
        Detects inverted ROS/OpenCV optical frame conventions automatically.
        """
        if not centroids_lidar:
            return "no_clusters"
        forward_count = 0
        in_front_cam = 0
        behind_cam = 0
        in_fov = 0
        for pt in centroids_lidar:
            if pt[0] > 0.5 or pt[2] > 0.5:
                forward_count += 1
            x_cam, y_cam, z_cam = _mat_vec_mul_3x3(self.extrinsics.rotation, pt)
            x_cam += self.extrinsics.translation[0]
            y_cam += self.extrinsics.translation[1]
            z_cam += self.extrinsics.translation[2]
            if z_cam > self.min_depth_m:
                in_front_cam += 1
                u = self.intrinsics.fx * (x_cam / z_cam) + self.intrinsics.cx
                v = self.intrinsics.fy * (y_cam / z_cam) + self.intrinsics.cy
                if 0.0 <= u < self.intrinsics.width and 0.0 <= v < self.intrinsics.height:
                    in_fov += 1
            else:
                behind_cam += 1

        if forward_count > 0 and in_front_cam == 0 and behind_cam > 0:
            return "error_points_behind_camera_check_optical_frame"
        if in_front_cam > 0 and in_fov == 0:
            return "warn_points_outside_image_fov"
        return "ok"

    @staticmethod
    def detections_from_msg(msg: Optional[CameraConeDetections]) -> List[DetectionRecord]:
        if msg is None:
            return []
        detections: List[DetectionRecord] = []
        for det in msg.detections:
            probs = _normalize_probs(det.class_probs)
            is_big = bool(getattr(det, "is_big_orange", False)) or (
                probs[3] > 0.35 and probs[3] >= max(probs[0], probs[1], probs[2])
            )
            poly_raw = getattr(det, "polygon_uv", None)
            poly = [(float(p[0]), float(p[1])) for p in poly_raw] if poly_raw else None
            detections.append(
                DetectionRecord(
                    xmin=float(det.xmin),
                    ymin=float(det.ymin),
                    xmax=float(det.xmax),
                    ymax=float(det.ymax),
                    class_probs=probs,
                    confidence=float(_clamp(det.confidence, 0.0, 1.0)),
                    is_big_orange=is_big,
                    polygon_uv=poly,
                )
            )
        return detections

    def estimate_cluster_semantics(
        self,
        centroid_lidar: Tuple[float, float, float],
        covariance: Sequence[float],
        detections: List[DetectionRecord],
        fallback_probs: Sequence[float],
        excluded_detection_indices: Optional[Sequence[int]] = None,
    ) -> ClusterSemanticEstimate:
        fallback = _normalize_probs(fallback_probs)
        samples = self._sample_cluster_points(centroid_lidar, covariance)
        excluded_set = set(int(i) for i in excluded_detection_indices) if excluded_detection_indices else set()

        acc = [0.0, 0.0, 0.0, 0.0]
        projected_samples = 0
        matched_samples = 0
        matched_weight_sum = 0.0
        best_confidence = 0.0
        residual_samples = 0
        residual_sum_px = 0.0
        residual_max_px = 0.0
        overall_best_det_idx: Optional[int] = None
        overall_best_det_weight = 0.0

        for point in samples:
            proj = project_lidar_point_to_image(
                point_lidar=point,
                intrinsics=self.intrinsics,
                extrinsics=self.extrinsics,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            if proj is None:
                continue
            projected_samples += 1
            u, v, depth = proj

            pixel_weight_sum = 0.0
            pixel_class_acc = [0.0, 0.0, 0.0, 0.0]
            best_det: Optional[DetectionRecord] = None
            best_det_idx: Optional[int] = None
            best_det_weight = 0.0
            for det_idx, det in enumerate(detections):
                if det_idx in excluded_set:
                    continue
                likelihood = self._detection_likelihood(det, u, v)
                if likelihood <= 0.0:
                    continue
                area = max(1.0, (det.xmax - det.xmin) * (det.ymax - det.ymin))
                det_weight = det.confidence * likelihood / math.sqrt(area)
                if det_weight <= 1e-9:
                    continue
                pixel_weight_sum += det_weight
                for i in range(4):
                    pixel_class_acc[i] += det_weight * det.class_probs[i]
                if det_weight > best_det_weight:
                    best_det_weight = det_weight
                    best_det = det
                    best_det_idx = det_idx

            if pixel_weight_sum <= 1e-9:
                continue
            matched_samples += 1

            depth_weight = _clamp(1.0 - depth / max(1.0, self.max_depth_m), 0.2, 1.0)
            weight = pixel_weight_sum * depth_weight
            matched_weight_sum += weight
            if best_det_weight * depth_weight > overall_best_det_weight:
                overall_best_det_weight = best_det_weight * depth_weight
                overall_best_det_idx = best_det_idx
            if best_det is not None:
                best_confidence = max(best_confidence, best_det.confidence)
                center_u = 0.5 * (best_det.xmin + best_det.xmax)
                center_v = 0.5 * (best_det.ymin + best_det.ymax)
                residual_px = math.hypot(u - center_u, v - center_v)
            else:
                residual_px = 0.0
            residual_samples += 1
            residual_sum_px += residual_px
            residual_max_px = max(residual_max_px, residual_px)
            for i in range(4):
                acc[i] += depth_weight * pixel_class_acc[i]

        if matched_weight_sum <= 1e-9:
            return ClusterSemanticEstimate(
                class_probs=fallback,
                semantic_weight=float(_clamp(self.no_match_semantic_weight, 0.0, 1.0)),
                diagnostics=ProjectionDiagnostics(
                    projected_samples=projected_samples,
                    matched_samples=matched_samples,
                    matched_weight_sum=matched_weight_sum,
                    best_confidence=best_confidence,
                    residual_samples=residual_samples,
                    residual_sum_px=residual_sum_px,
                    residual_max_px=residual_max_px,
                    matched_detection_idx=None,
                ),
            )

        painted_probs = [v / matched_weight_sum for v in acc]
        painted_probs = _normalize_probs(painted_probs)
        coverage = (
            matched_samples / max(1.0, float(projected_samples))
            if projected_samples > 0
            else 0.0
        )
        semantic_weight = _clamp(
            self.min_semantic_weight + (1.0 - self.min_semantic_weight) * coverage * best_confidence,
            self.min_semantic_weight,
            1.0,
        )

        return ClusterSemanticEstimate(
            class_probs=painted_probs,
            semantic_weight=float(semantic_weight),
            diagnostics=ProjectionDiagnostics(
                projected_samples=projected_samples,
                matched_samples=matched_samples,
                matched_weight_sum=matched_weight_sum,
                best_confidence=best_confidence,
                residual_samples=residual_samples,
                residual_sum_px=residual_sum_px,
                residual_max_px=residual_max_px,
                matched_detection_idx=overall_best_det_idx,
            ),
        )

    def estimate_batch_cluster_semantics(
        self,
        clusters_lidar: Sequence[Tuple[Tuple[float, float, float], Sequence[float]]],
        detections: List[DetectionRecord],
        fallback_probs: Sequence[float],
    ) -> List[ClusterSemanticEstimate]:
        """
        Global optimal 2D-3D bipartite semantic assignment across all LiDAR clusters
        and 2D camera detections using Hungarian optimization over projected frustum
        likelihood and normalized pixel residual cost.
        """
        n_clusters = len(clusters_lidar)
        if n_clusters == 0:
            return []
        n_dets = len(detections)
        if n_dets == 0:
            return [
                self.estimate_cluster_semantics(
                    centroid_lidar=c_pos,
                    covariance=c_cov,
                    detections=[],
                    fallback_probs=fallback_probs,
                )
                for c_pos, c_cov in clusters_lidar
            ]

        cpp_lib = get_cpp_kernels()
        if cpp_lib is not None:
            fb_norm = _normalize_probs(fallback_probs)
            c_intr = CameraIntrinsicsC(
                float(self.intrinsics.fx),
                float(self.intrinsics.fy),
                float(self.intrinsics.cx),
                float(self.intrinsics.cy),
                int(self.intrinsics.width),
                int(self.intrinsics.height),
                float(self.intrinsics.k1),
                float(self.intrinsics.k2),
                float(self.intrinsics.p1),
                float(self.intrinsics.p2),
                float(self.intrinsics.k3),
            )
            flat_r = [float(v) for row in self.extrinsics.rotation for v in row]
            c_ext = ExtrinsicsC(
                (ctypes.c_double * 9)(*flat_r),
                (ctypes.c_double * 3)(*[float(v) for v in self.extrinsics.translation]),
                float(self.min_depth_m),
                float(self.max_depth_m),
                float(self.sample_radius_scale),
                float(self.min_semantic_weight),
                float(self.no_match_semantic_weight),
                1 if self.soft_likelihood_enabled else 0,
                float(self.soft_likelihood_sigma_scale),
                float(self.soft_likelihood_min_value),
                float(self.soft_likelihood_outside_ratio),
                1 if self.enable_frustum_hull_sampling else 0,
                1 if getattr(self, "enable_triangular_cone_mask", False) else 0,
                float(getattr(self, "cone_mask_apex_width_ratio", 0.18)),
            )
            c_clusters = (ClusterInputC * n_clusters)()
            for i, (c_pos, c_cov) in enumerate(clusters_lidar):
                if len(c_cov) != 9 or all(abs(float(v)) < 1e-12 for v in c_cov):
                    cxx, cyy, czz = 0.05, 0.05, 0.1
                else:
                    cxx, cyy, czz = float(c_cov[0]), float(c_cov[4]), float(c_cov[8])
                c_clusters[i] = ClusterInputC(
                    float(c_pos[0]), float(c_pos[1]), float(c_pos[2]), cxx, cyy, czz
                )
            c_dets = (Detection2DC * n_dets)()
            for i, det in enumerate(detections):
                dp = _normalize_probs(det.class_probs)
                poly = getattr(det, "polygon_uv", None) or []
                poly_cnt = min(16, len(poly)) if len(poly) >= 3 else 0
                pu_arr = (ctypes.c_double * 16)(*[float(poly[k][0]) if k < poly_cnt else 0.0 for k in range(16)])
                pv_arr = (ctypes.c_double * 16)(*[float(poly[k][1]) if k < poly_cnt else 0.0 for k in range(16)])
                c_dets[i] = Detection2DC(
                    float(det.xmin),
                    float(det.ymin),
                    float(det.xmax),
                    float(det.ymax),
                    (ctypes.c_double * 4)(dp[0], dp[1], dp[2], dp[3]),
                    float(det.confidence),
                    int(poly_cnt),
                    pu_arr,
                    pv_arr,
                    1 if getattr(det, "is_big_orange", False) else 0,
                )
            c_fb = (ctypes.c_double * 4)(fb_norm[0], fb_norm[1], fb_norm[2], fb_norm[3])
            c_out = (ClusterSemanticC * n_clusters)()
            n_ret = cpp_lib.cpp_estimate_batch_semantics(
                n_clusters,
                c_clusters,
                n_dets,
                c_dets,
                ctypes.byref(c_intr),
                ctypes.byref(c_ext),
                c_fb,
                c_out,
            )
            if n_ret == n_clusters:
                out_list: List[ClusterSemanticEstimate] = []
                for i in range(n_clusters):
                    r = c_out[i]
                    matched_idx = None if r.matched_detection_idx < 0 else int(r.matched_detection_idx)
                    out_list.append(
                        ClusterSemanticEstimate(
                            class_probs=[
                                float(r.class_probs[0]),
                                float(r.class_probs[1]),
                                float(r.class_probs[2]),
                                float(r.class_probs[3]),
                            ],
                            semantic_weight=float(r.semantic_weight),
                            diagnostics=ProjectionDiagnostics(
                                projected_samples=int(r.projected_samples),
                                matched_samples=int(r.matched_samples),
                                matched_weight_sum=float(r.matched_weight_sum),
                                best_confidence=float(r.best_confidence),
                                residual_samples=int(r.residual_samples),
                                residual_sum_px=float(r.residual_sum_px),
                                residual_max_px=float(r.residual_max_px),
                                matched_detection_idx=matched_idx,
                            ),
                        )
                    )
                return out_list

        disallowed_cost = 1e6
        pair_scores: Dict[Tuple[int, int], float] = {}
        for c_idx, (c_pos, c_cov) in enumerate(clusters_lidar):
            samples = self._sample_cluster_points(c_pos, c_cov)
            proj_pts: List[Tuple[float, float, float]] = []
            for pt in samples:
                pr = project_lidar_point_to_image(
                    point_lidar=pt,
                    intrinsics=self.intrinsics,
                    extrinsics=self.extrinsics,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                )
                if pr is not None:
                    proj_pts.append(pr)
            if not proj_pts:
                continue

            for d_idx, det in enumerate(detections):
                area = max(1.0, (det.xmax - det.xmin) * (det.ymax - det.ymin))
                inv_sqrt_area = 1.0 / math.sqrt(area)
                center_u = 0.5 * (det.xmin + det.xmax)
                center_v = 0.5 * (det.ymin + det.ymax)
                diag_px = max(4.0, math.hypot(det.xmax - det.xmin, det.ymax - det.ymin))
                w_sum = 0.0
                res_norm_sum = 0.0
                matched_cnt = 0
                for u, v, depth in proj_pts:
                    lk = self._detection_likelihood(det, u, v)
                    if lk <= 0.0:
                        continue
                    dw = _clamp(1.0 - depth / max(1.0, self.max_depth_m), 0.2, 1.0)
                    w_sum += det.confidence * lk * inv_sqrt_area * dw
                    res_norm_sum += math.hypot(u - center_u, v - center_v) / diag_px
                    matched_cnt += 1
                if matched_cnt > 0 and w_sum > 1e-9:
                    mean_res_norm = res_norm_sum / float(matched_cnt)
                    # Lower cost for higher likelihood weight and tighter normalized pixel residual
                    cost = max(0.01, 10.0 / (1.0 + 25.0 * w_sum) + 0.5 * mean_res_norm)
                    pair_scores[(c_idx, d_idx)] = cost

        assigned_det_for_cluster: Dict[int, int] = {}
        if _HAS_SCIPY and n_clusters > 1 and n_dets > 1 and pair_scores:
            cost_mat = np.full((n_clusters, n_dets), disallowed_cost, dtype=np.float64)
            for (c_idx, d_idx), cost in pair_scores.items():
                cost_mat[c_idx, d_idx] = float(cost)
            r_ind, c_ind = linear_sum_assignment(cost_mat)
            for r_i, c_i in zip(r_ind, c_ind):
                if (int(r_i), int(c_i)) in pair_scores:
                    assigned_det_for_cluster[int(r_i)] = int(c_i)
        elif pair_scores:
            used_c = set()
            used_d = set()
            for (c_idx, d_idx), _cost in sorted(pair_scores.items(), key=lambda kv: kv[1]):
                if c_idx in used_c or d_idx in used_d:
                    continue
                used_c.add(c_idx)
                used_d.add(d_idx)
                assigned_det_for_cluster[c_idx] = d_idx

        results: List[ClusterSemanticEstimate] = []
        all_det_indices = set(range(n_dets))
        for c_idx, (c_pos, c_cov) in enumerate(clusters_lidar):
            target_det = assigned_det_for_cluster.get(c_idx)
            if target_det is not None:
                excluded = list(all_det_indices - {target_det})
            else:
                excluded = list(all_det_indices)
            est = self.estimate_cluster_semantics(
                centroid_lidar=c_pos,
                covariance=c_cov,
                detections=detections,
                fallback_probs=fallback_probs,
                excluded_detection_indices=excluded,
            )
            results.append(est)
        return results

    def recover_unmatched_detections(
        self,
        detections: Sequence[DetectionRecord],
        matched_detection_indices: Sequence[int],
        existing_positions: Sequence[Tuple[float, float, float]],
        cone_height_m: float = 0.32,
        big_orange_height_m: float = 0.505,
        min_confidence: float = 0.55,
        min_bbox_height_px: float = 8.0,
        min_depth_m: float = 3.0,
        max_depth_m: float = 35.0,
        dedup_radius_m: float = 1.2,
        range_std_fraction: float = 0.14,
        bearing_std_fraction: float = 0.025,
        enable_ground_plane_depth_fusion: bool = False,
        ground_z_base_m: float = -0.12,
        ground_slope_rad: float = 0.0,
        stereo_depth: Optional[StereoDepthFrame] = None,
        enable_stereo_depth_fusion: bool = True,
    ) -> List[FrustumRecoveredCone]:
        """
        Bidirectional Camera -> 3D Frustum recovery for cones missed by LiDAR clustering
        (e.g. long-range 18-35m cones where LiDAR scan lines are sparse).
        Computes ray-aligned anisotropic covariance (tight bearing, wider range),
        supports FSG 505mm big-orange gate cones, and fuses ZED stereo depth/disparity,
        ground-plane homography intersection, and bbox-height depth via inverse-variance weighting.
        """
        matched_set = set(matched_detection_indices)
        recovered: List[FrustumRecoveredCone] = []
        fx = max(1e-6, float(self.intrinsics.fx))
        fy = max(1e-6, float(self.intrinsics.fy))
        r = self.extrinsics.rotation
        t = self.extrinsics.translation
        cam_orig_x = -(r[0][0] * float(t[0]) + r[1][0] * float(t[1]) + r[2][0] * float(t[2]))
        cam_orig_z = -(r[0][2] * float(t[0]) + r[1][2] * float(t[1]) + r[2][2] * float(t[2]))
        tan_slope = math.tan(float(ground_slope_rad))

        for idx, det in enumerate(detections):
            if idx in matched_set:
                continue
            if det.confidence < min_confidence:
                continue
            h_px = max(0.0, float(det.ymax - det.ymin))
            if h_px < min_bbox_height_px:
                continue

            eff_height_m = (
                float(big_orange_height_m)
                if getattr(det, "is_big_orange", False)
                else float(cone_height_m)
            )
            depth_bbox = float(eff_height_m * fy / max(1e-3, h_px))
            sigma_bbox = max(0.10, 0.14 * depth_bbox)

            poly = getattr(det, "polygon_uv", None)
            if poly and len(poly) >= 3:
                u = sum(p[0] for p in poly) / float(len(poly))
                v = sum(p[1] for p in poly) / float(len(poly))
            else:
                u = 0.5 * (float(det.xmin) + float(det.xmax))
                v = 0.5 * (float(det.ymin) + float(det.ymax))
            xd = (u - float(self.intrinsics.cx)) / fx
            yd = (v - float(self.intrinsics.cy)) / fy
            xn, yn = self.intrinsics.undistort_normalized(xd, yd)

            inv_var_sum = 1.0 / (sigma_bbox * sigma_bbox)
            weighted_depth_sum = depth_bbox * inv_var_sum

            if enable_ground_plane_depth_fusion:
                # Back-project bottom-center pixel (u, ymax) ray and intersect with local ground plane
                yd_bot = (float(det.ymax) - float(self.intrinsics.cy)) / fy
                xn_b, yn_b = self.intrinsics.undistort_normalized(xd, yd_bot)
                dir_x = r[0][0] * xn_b + r[1][0] * yn_b + r[2][0] * 1.0
                dir_z = r[0][2] * xn_b + r[1][2] * yn_b + r[2][2] * 1.0
                denom_g = dir_z - dir_x * tan_slope
                if denom_g < -1e-3:
                    s_ground = (
                        float(ground_z_base_m)
                        + cam_orig_x * tan_slope
                        - cam_orig_z
                    ) / denom_g
                    if min_depth_m <= s_ground <= max_depth_m and abs(s_ground - depth_bbox) <= 0.35 * depth_bbox:
                        sigma_gnd = max(0.08, 0.09 * s_ground)
                        w_gnd = 1.0 / (sigma_gnd * sigma_gnd)
                        inv_var_sum += w_gnd
                        weighted_depth_sum += s_ground * w_gnd

            used_stereo = False
            if enable_stereo_depth_fusion and stereo_depth is not None:
                stereo_est = sample_stereo_depth_for_detection(det, stereo_depth, self.intrinsics)
                if stereo_est is not None:
                    z_stereo, sigma_stereo = stereo_est
                    prior_depth = weighted_depth_sum / inv_var_sum
                    if (
                        min_depth_m <= z_stereo <= max_depth_m
                        and abs(z_stereo - prior_depth) <= 0.45 * max(1.0, prior_depth)
                    ):
                        w_stereo = 1.0 / (sigma_stereo * sigma_stereo)
                        inv_var_sum += w_stereo
                        weighted_depth_sum += z_stereo * w_stereo
                        used_stereo = True

            depth = weighted_depth_sum / inv_var_sum
            if depth < min_depth_m or depth > max_depth_m:
                continue

            x_cam = xn * depth
            y_cam = yn * depth
            z_cam = depth

            # Back-project to LiDAR/ego frame: p_lidar = R^T * (p_cam - t)
            x0 = x_cam - float(t[0])
            y0 = y_cam - float(t[1])
            z0 = z_cam - float(t[2])
            x_l = r[0][0] * x0 + r[1][0] * y0 + r[2][0] * z0
            y_l = r[0][1] * x0 + r[1][1] * y0 + r[2][1] * z0
            z_l = r[0][2] * x0 + r[1][2] * y0 + r[2][2] * z0

            # Deduplicate against existing 3D clusters and already-recovered frustums
            too_close = False
            for ex, ey, _ in existing_positions:
                if math.hypot(x_l - ex, y_l - ey) < dedup_radius_m:
                    too_close = True
                    break
            if too_close:
                continue
            for prev in recovered:
                if math.hypot(x_l - prev.position[0], y_l - prev.position[1]) < dedup_radius_m:
                    too_close = True
                    break
            if too_close:
                continue

            # Anisotropic ray-aligned covariance: R(psi) diag(var_range, var_bearing) R(psi)^T
            psi = math.atan2(y_l, x_l)
            c_p, s_p = math.cos(psi), math.sin(psi)
            sigma_r = max(0.12, min(float(range_std_fraction) * depth, 1.0 / math.sqrt(inv_var_sum)))
            sigma_b = max(0.05, bearing_std_fraction * depth)
            var_r = sigma_r * sigma_r
            var_b = sigma_b * sigma_b

            cov_xx = c_p * c_p * var_r + s_p * s_p * var_b
            cov_yy = s_p * s_p * var_r + c_p * c_p * var_b
            cov_xy = c_p * s_p * (var_r - var_b)
            var_z = max(0.10, 0.25 * var_r)

            recovered.append(
                FrustumRecoveredCone(
                    position=(float(x_l), float(y_l), float(z_l)),
                    covariance=[
                        float(cov_xx),
                        float(cov_xy),
                        0.0,
                        float(cov_xy),
                        float(cov_yy),
                        0.0,
                        0.0,
                        0.0,
                        float(var_z),
                    ],
                    class_probs=_normalize_probs(det.class_probs),
                    confidence=float(det.confidence),
                    depth_m=float(depth),
                    detection_idx=idx,
                    used_stereo=used_stereo,
                )
            )

        return recovered

    def _sample_cluster_points(
        self, centroid_lidar: Tuple[float, float, float], covariance: Sequence[float]
    ) -> List[Tuple[float, float, float]]:
        x, y, z = centroid_lidar
        cov = list(covariance)
        if len(cov) != 9 or all(abs(v) < 1e-12 for v in cov):
            cov = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.1]
        var_x = max(1e-4, cov[0])
        var_y = max(1e-4, cov[4])
        var_z = max(1e-4, cov[8])
        radius = self.sample_radius_scale * max(0.05, math.sqrt(max(var_x, var_y)))
        z_off = 0.5 * math.sqrt(var_z)
        pts = [
            (x, y, z),
            (x + radius, y, z),
            (x - radius, y, z),
            (x, y + radius, z),
            (x, y - radius, z),
            (x, y, z + z_off),
            (x, y, z - z_off),
        ]
        if getattr(self, "enable_frustum_hull_sampling", False):
            # Physical FSG 3D cone frustum wireframe points (apex + mid-cone lateral edges + base skirt)
            cone_r = 0.095
            pts.extend(
                [
                    (x, y, z + 0.16),
                    (x, y + cone_r, z - 0.06),
                    (x, y - cone_r, z - 0.06),
                    (x, y + 0.5 * cone_r, z + 0.08),
                    (x, y - 0.5 * cone_r, z + 0.08),
                ]
            )
        return pts

    @staticmethod
    def _best_detection_for_pixel(
        detections: List[DetectionRecord], u: float, v: float
    ) -> Optional[DetectionRecord]:
        best: Optional[DetectionRecord] = None
        best_score = -1.0
        for det in detections:
            if u < det.xmin or u > det.xmax or v < det.ymin or v > det.ymax:
                continue
            area = max(1.0, (det.xmax - det.xmin) * (det.ymax - det.ymin))
            score = det.confidence / area
            if score > best_score:
                best_score = score
                best = det
        return best

    def _detection_likelihood(self, det: DetectionRecord, u: float, v: float) -> float:
        width = max(1.0, det.xmax - det.xmin)
        height = max(1.0, det.ymax - det.ymin)
        cx = 0.5 * (det.xmin + det.xmax)
        cy = 0.5 * (det.ymin + det.ymax)
        half_w = 0.5 * width
        half_h = 0.5 * height

        poly = getattr(det, "polygon_uv", None)
        if poly and len(poly) >= 3:
            poly_use = poly[:16]
            inside = _point_in_polygon(poly_use, u, v)
            if not self.soft_likelihood_enabled:
                return 1.0 if inside else 0.0
            n = len(poly_use)
            p_cu = sum(p[0] for p in poly_use) / float(n)
            p_cv = sum(p[1] for p in poly_use) / float(n)
            us = [p[0] for p in poly_use]
            vs = [p[1] for p in poly_use]
            pw = max(1.0, max(us) - min(us))
            ph = max(1.0, max(vs) - min(vs))
            if not inside:
                d_edge = _min_dist_to_polygon_edges(poly_use, u, v)
                max_outside = max(
                    2.0, max(0.0, self.soft_likelihood_outside_ratio - 1.0) * 0.5 * min(pw, ph)
                )
                if d_edge > max_outside:
                    return 0.0
            sigma_u = max(1.0, self.soft_likelihood_sigma_scale * pw)
            sigma_v = max(1.0, self.soft_likelihood_sigma_scale * ph)
            du = (u - p_cu) / sigma_u
            dv = (v - p_cv) / sigma_v
            value = math.exp(-0.5 * (du * du + dv * dv))
            if not inside:
                value *= 0.65
            if value < self.soft_likelihood_min_value:
                return 0.0
            return float(_clamp(value, 0.0, 1.0))

        if getattr(self, "enable_triangular_cone_mask", False):
            apex_ratio = _clamp(getattr(self, "cone_mask_apex_width_ratio", 0.18), 0.0, 1.0)
            t_vert = _clamp((v - det.ymin) / height, 0.0, 1.0)
            half_w_t = max(0.5, half_w * (apex_ratio + (1.0 - apex_ratio) * t_vert))
            if not self.soft_likelihood_enabled:
                if v < det.ymin or v > det.ymax or abs(u - cx) > half_w_t:
                    return 0.0
                return 1.0
            outside_gate_w = max(1.0, self.soft_likelihood_outside_ratio * half_w_t)
            outside_gate_h = max(1.0, self.soft_likelihood_outside_ratio * half_h)
            if abs(u - cx) > outside_gate_w or abs(v - cy) > outside_gate_h:
                return 0.0
            sigma_u = max(1.0, self.soft_likelihood_sigma_scale * (2.0 * half_w_t))
            sigma_v = max(1.0, self.soft_likelihood_sigma_scale * height)
            du = (u - cx) / sigma_u
            dv = (v - cy) / sigma_v
            value = math.exp(-0.5 * (du * du + dv * dv))
            if value < self.soft_likelihood_min_value:
                return 0.0
            return float(_clamp(value, 0.0, 1.0))

        if not self.soft_likelihood_enabled:
            if u < det.xmin or u > det.xmax or v < det.ymin or v > det.ymax:
                return 0.0
            return 1.0

        outside_gate_w = max(1.0, self.soft_likelihood_outside_ratio * half_w)
        outside_gate_h = max(1.0, self.soft_likelihood_outside_ratio * half_h)
        if abs(u - cx) > outside_gate_w or abs(v - cy) > outside_gate_h:
            return 0.0

        sigma_u = max(1.0, self.soft_likelihood_sigma_scale * width)
        sigma_v = max(1.0, self.soft_likelihood_sigma_scale * height)
        du = (u - cx) / sigma_u
        dv = (v - cy) / sigma_v
        value = math.exp(-0.5 * (du * du + dv * dv))
        if value < self.soft_likelihood_min_value:
            return 0.0
        return float(_clamp(value, 0.0, 1.0))


@dataclass
class CameraSensorProfile:
    """Calibration and topic configuration for a single camera sensor in an N-camera rig."""

    camera_id: str
    topic: str
    frame_id: str
    projector: EarlyFusionProjector
    use_tf_extrinsics: bool = False
    camera_info_topic: str = ""
    depth_topic: str = ""


class MultiCameraEarlyFusion:
    """Multi-camera semantic painter and cross-camera frustum recovery coordinator."""

    def __init__(self, profiles: Optional[Dict[str, CameraSensorProfile]] = None) -> None:
        self.profiles: Dict[str, CameraSensorProfile] = dict(profiles) if profiles else {}
        self.last_matched_indices_by_cam: Dict[str, List[int]] = {}

    def register_camera(self, profile: CameraSensorProfile) -> None:
        self.profiles[profile.camera_id] = profile

    def estimate_batch_multi_camera_semantics(
        self,
        clusters_lidar_by_cam: Dict[
            str, Sequence[Tuple[Tuple[float, float, float], Sequence[float]]]
        ],
        detections_by_cam: Dict[str, List[DetectionRecord]],
        fallback_probs: Sequence[float],
    ) -> List[ClusterSemanticEstimate]:
        """Fuse per-camera Hungarian semantic estimates across N cameras per 3D cluster."""
        fallback = _normalize_probs(fallback_probs)
        n_clusters = 0
        for clusters in clusters_lidar_by_cam.values():
            n_clusters = len(clusters)
            break
        if n_clusters == 0:
            self.last_matched_indices_by_cam = {}
            return []

        per_cam_estimates: Dict[str, List[ClusterSemanticEstimate]] = {}
        matched_by_cam: Dict[str, List[int]] = {}
        for cam_id, profile in self.profiles.items():
            dets = detections_by_cam.get(cam_id, [])
            clusters = clusters_lidar_by_cam.get(cam_id, [])
            if not clusters or not dets:
                continue
            ests = profile.projector.estimate_batch_cluster_semantics(
                clusters_lidar=clusters,
                detections=dets,
                fallback_probs=fallback,
            )
            per_cam_estimates[cam_id] = ests
            matched_by_cam[cam_id] = [
                int(est.diagnostics.matched_detection_idx)
                for est in ests
                if est.diagnostics.matched_detection_idx is not None
            ]

        self.last_matched_indices_by_cam = matched_by_cam
        if not per_cam_estimates:
            return [
                ClusterSemanticEstimate(
                    class_probs=fallback,
                    semantic_weight=0.6,
                    diagnostics=ProjectionDiagnostics(
                        projected_samples=0,
                        matched_samples=0,
                        matched_weight_sum=0.0,
                        best_confidence=0.0,
                        residual_samples=0,
                        residual_sum_px=0.0,
                        residual_max_px=0.0,
                        matched_detection_idx=None,
                    ),
                )
                for _ in range(n_clusters)
            ]

        fused: List[ClusterSemanticEstimate] = []
        for c_idx in range(n_clusters):
            matched_ests: List[ClusterSemanticEstimate] = []
            unmatched_ests: List[ClusterSemanticEstimate] = []
            for ests in per_cam_estimates.values():
                if c_idx < len(ests):
                    est = ests[c_idx]
                    if est.diagnostics.matched_detection_idx is not None:
                        matched_ests.append(est)
                    else:
                        unmatched_ests.append(est)

            if not matched_ests:
                # Pick the estimate with the highest projected sample count or first available
                best_unmatched = max(
                    unmatched_ests,
                    key=lambda e: (e.diagnostics.projected_samples, e.semantic_weight),
                )
                fused.append(best_unmatched)
                continue

            if len(matched_ests) == 1:
                fused.append(matched_ests[0])
                continue

            # Weighted combination across multiple cameras seeing the same 3D cone
            acc = [0.0, 0.0, 0.0, 0.0]
            w_total = 0.0
            proj_samples = 0
            matched_samples = 0
            weight_sum = 0.0
            best_conf = 0.0
            res_samples = 0
            res_sum = 0.0
            res_max = 0.0
            best_sem_w = 0.0
            best_det_idx: Optional[int] = None

            for est in matched_ests:
                w = max(1e-4, float(est.semantic_weight) * max(0.1, float(est.diagnostics.best_confidence)))
                for k in range(4):
                    acc[k] += w * float(est.class_probs[k])
                w_total += w
                proj_samples += int(est.diagnostics.projected_samples)
                matched_samples += int(est.diagnostics.matched_samples)
                weight_sum += float(est.diagnostics.matched_weight_sum)
                if est.diagnostics.best_confidence > best_conf:
                    best_conf = float(est.diagnostics.best_confidence)
                    best_det_idx = est.diagnostics.matched_detection_idx
                res_samples += int(est.diagnostics.residual_samples)
                res_sum += float(est.diagnostics.residual_sum_px)
                res_max = max(res_max, float(est.diagnostics.residual_max_px))
                best_sem_w = max(best_sem_w, float(est.semantic_weight))

            fused_probs = _normalize_probs([v / max(1e-9, w_total) for v in acc])
            fused.append(
                ClusterSemanticEstimate(
                    class_probs=fused_probs,
                    semantic_weight=float(_clamp(best_sem_w, 0.0, 1.0)),
                    diagnostics=ProjectionDiagnostics(
                        projected_samples=proj_samples,
                        matched_samples=matched_samples,
                        matched_weight_sum=weight_sum,
                        best_confidence=best_conf,
                        residual_samples=res_samples,
                        residual_sum_px=res_sum,
                        residual_max_px=res_max,
                        matched_detection_idx=best_det_idx,
                    ),
                )
            )
        return fused

    def recover_multi_camera_unmatched_detections(
        self,
        detections_by_cam: Dict[str, List[DetectionRecord]],
        matched_indices_by_cam: Dict[str, Sequence[int]],
        existing_positions: Sequence[Tuple[float, float, float]],
        cone_height_m: float = 0.32,
        big_orange_height_m: float = 0.505,
        min_confidence: float = 0.55,
        min_bbox_height_px: float = 8.0,
        min_depth_m: float = 3.0,
        max_depth_m: float = 35.0,
        dedup_radius_m: float = 1.2,
        enable_ground_plane_depth_fusion: bool = False,
        ground_z_base_m: float = -0.12,
        ground_slope_rad: float = 0.0,
        stereo_depth_by_cam: Optional[Dict[str, StereoDepthFrame]] = None,
        enable_stereo_depth_fusion: bool = True,
    ) -> Dict[str, List[FrustumRecoveredCone]]:
        """Recover unmatched 2D detections across N cameras while deduplicating across views."""
        occupied: List[Tuple[float, float, float]] = list(existing_positions)
        recovered_by_cam: Dict[str, List[FrustumRecoveredCone]] = {}
        for cam_id, profile in self.profiles.items():
            dets = detections_by_cam.get(cam_id, [])
            if not dets:
                continue
            matched = matched_indices_by_cam.get(cam_id, [])
            stereo_frame = stereo_depth_by_cam.get(cam_id) if stereo_depth_by_cam else None
            rec_list = profile.projector.recover_unmatched_detections(
                detections=dets,
                matched_detection_indices=matched,
                existing_positions=occupied,
                cone_height_m=cone_height_m,
                big_orange_height_m=big_orange_height_m,
                min_confidence=min_confidence,
                min_bbox_height_px=min_bbox_height_px,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                dedup_radius_m=dedup_radius_m,
                enable_ground_plane_depth_fusion=enable_ground_plane_depth_fusion,
                ground_z_base_m=ground_z_base_m,
                ground_slope_rad=ground_slope_rad,
                stereo_depth=stereo_frame,
                enable_stereo_depth_fusion=enable_stereo_depth_fusion,
            )
            if rec_list:
                recovered_by_cam[cam_id] = rec_list
                for rc in rec_list:
                    occupied.append(rc.position)
        return recovered_by_cam
