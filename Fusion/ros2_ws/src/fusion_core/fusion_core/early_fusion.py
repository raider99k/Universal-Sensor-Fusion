from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

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
            detections.append(
                DetectionRecord(
                    xmin=float(det.xmin),
                    ymin=float(det.ymin),
                    xmax=float(det.xmax),
                    ymax=float(det.ymax),
                    class_probs=_normalize_probs(det.class_probs),
                    confidence=float(_clamp(det.confidence, 0.0, 1.0)),
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
    ) -> List[FrustumRecoveredCone]:
        """
        Bidirectional Camera -> 3D Frustum recovery for cones missed by LiDAR clustering
        (e.g. long-range 18-35m cones where LiDAR scan lines are sparse).
        Computes ray-aligned anisotropic covariance (tight bearing, wider range),
        supports FSG 505mm big-orange gate cones, and optionally fuses ground-plane
        homography intersection with bbox-height depth via inverse-variance weighting.
        """
        matched_set = set(matched_detection_indices)
        recovered: List[FrustumRecoveredCone] = []
        fx = max(1e-6, float(self.intrinsics.fx))
        fy = max(1e-6, float(self.intrinsics.fy))
        r = self.extrinsics.rotation
        t = self.extrinsics.translation
        cam_orig_x = -(r[0][0] * float(t[0]) + r[1][0] * float(t[1]) + r[2][0] * float(t[2]))
        cam_orig_y = -(r[0][1] * float(t[0]) + r[1][1] * float(t[1]) + r[2][1] * float(t[2]))
        cam_orig_z = -(r[0][2] * float(t[0]) + r[1][2] * float(t[1]) + r[2][2] * float(t[2]))

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
            depth = depth_bbox
            eff_range_std_fraction = float(range_std_fraction)

            u = 0.5 * (float(det.xmin) + float(det.xmax))
            v = 0.5 * (float(det.ymin) + float(det.ymax))
            xd = (u - float(self.intrinsics.cx)) / fx
            yd = (v - float(self.intrinsics.cy)) / fy
            xn, yn = self.intrinsics.undistort_normalized(xd, yd)

            if enable_ground_plane_depth_fusion:
                # Back-project bottom-center pixel (u, ymax) ray and intersect with local ground plane
                yd_bot = (float(det.ymax) - float(self.intrinsics.cy)) / fy
                xn_b, yn_b = self.intrinsics.undistort_normalized(xd, yd_bot)
                dir_x = r[0][0] * xn_b + r[1][0] * yn_b + r[2][0] * 1.0
                dir_y = r[0][1] * xn_b + r[1][1] * yn_b + r[2][1] * 1.0
                dir_z = r[0][2] * xn_b + r[1][2] * yn_b + r[2][2] * 1.0
                denom_g = dir_z - dir_x * math.tan(float(ground_slope_rad))
                if denom_g < -1e-3:
                    s_ground = (
                        float(ground_z_base_m)
                        + cam_orig_x * math.tan(float(ground_slope_rad))
                        - cam_orig_z
                    ) / denom_g
                    if min_depth_m <= s_ground <= max_depth_m and abs(s_ground - depth_bbox) <= 0.35 * depth_bbox:
                        # Inverse-variance fusion of bbox-height depth and ground-ray homography depth
                        var_bbox = (0.14 * depth_bbox) ** 2
                        var_gnd = (0.09 * s_ground) ** 2
                        depth = (var_gnd * depth_bbox + var_bbox * s_ground) / (var_bbox + var_gnd)
                        eff_range_std_fraction = float(range_std_fraction) * 0.78

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
            sigma_r = max(0.20, eff_range_std_fraction * depth)
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
