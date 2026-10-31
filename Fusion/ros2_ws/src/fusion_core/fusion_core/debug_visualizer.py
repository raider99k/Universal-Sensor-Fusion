from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fusion_core.early_fusion import (
    CameraIntrinsics,
    DetectionRecord,
    LidarToCameraExtrinsics,
    project_lidar_point_to_image,
)
from fusion_core.ekf_tracker import (
    CorridorEstimate,
    TRACK_CONFIRMED,
    TrackEstimate,
)


# Standard Formula Student color palette (RGBA in [0, 1])
CLASS_RGBA: Tuple[Tuple[float, float, float, float], ...] = (
    (0.12, 0.47, 0.96, 0.92),  # 0: Blue cone
    (0.98, 0.85, 0.10, 0.92),  # 1: Yellow cone
    (0.98, 0.45, 0.08, 0.92),  # 2: Small orange cone
    (1.00, 0.22, 0.05, 0.96),  # 3: Big orange start/finish cone
)
CLASS_NAMES: Tuple[str, ...] = ("BLUE", "YELLOW", "ORANGE", "BIG_ORANGE")


@dataclass
class DebugMarkerDescriptor:
    """ROS-agnostic descriptor for RViz2 / Foxglove 3D Marker visualization."""

    ns: str
    marker_id: int
    marker_type: str  # "CYLINDER", "ELLIPSE", "TEXT_VIEW_FACING", "LINE_STRIP"
    frame_id: str
    stamp_sec: float
    x: float
    y: float
    z: float
    yaw_rad: float
    scale_x: float
    scale_y: float
    scale_z: float
    rgba: Tuple[float, float, float, float]
    text: str = ""
    points_xyz: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class ReprojectionAnnotation:
    """2D camera-plane reprojection diagnostic annotation for Foxglove / debug image overlay."""

    cluster_index: int
    u_proj: float
    v_proj: float
    depth_m: float
    frustum_polygon_uv: List[Tuple[float, float]]
    matched_detection_index: int
    residual_px: float
    class_label: str


def compute_covariance_ellipse_2sigma(
    cov_xx: float, cov_xy: float, cov_yy: float
) -> Tuple[float, float, float]:
    """Return (major_diameter_m, minor_diameter_m, yaw_rad) for a 2-sigma 2D covariance ellipse."""
    trace = float(cov_xx + cov_yy)
    diff = float(cov_xx - cov_yy)
    disc = math.sqrt(max(0.0, diff * diff + 4.0 * cov_xy * cov_xy))
    eig1 = max(1e-6, 0.5 * (trace + disc))
    eig2 = max(1e-6, 0.5 * (trace - disc))
    yaw = 0.5 * math.atan2(2.0 * cov_xy, diff)
    # 2-sigma diameter = 2 * 2 * sqrt(eig) = 4 * sqrt(eig)
    return (
        float(4.0 * math.sqrt(eig1)),
        float(4.0 * math.sqrt(eig2)),
        float(yaw),
    )


def build_scene_marker_descriptors(
    tracks: Sequence[TrackEstimate],
    corridor: Optional[CorridorEstimate] = None,
    frame_id: str = "base_link",
    stamp_sec: float = 0.0,
    confirmed_only: bool = True,
) -> List[DebugMarkerDescriptor]:
    """Build 3D RViz2 / Foxglove marker descriptors for cone tracks, 2-sigma ellipses, and clothoid corridor."""
    markers: List[DebugMarkerDescriptor] = []

    for tr in tracks:
        if confirmed_only and tr.status != TRACK_CONFIRMED:
            continue
        x = float(tr.state[0])
        y = float(tr.state[1])
        z = float(tr.z)
        probs = list(tr.class_probs) if tr.class_probs else [0.25, 0.25, 0.25, 0.25]
        cls_idx = int(max(range(min(4, len(probs))), key=lambda i: probs[i]))
        max_p = float(probs[cls_idx])

        is_big = cls_idx == 3
        cone_h = 0.505 if is_big else 0.325
        cone_d = 0.285 if is_big else 0.228
        base_r, base_g, base_b, base_a = CLASS_RGBA[cls_idx]
        alpha = max(0.35, min(1.0, base_a * float(tr.quality)))

        # 1. 3D Cone Cylinder Marker
        markers.append(
            DebugMarkerDescriptor(
                ns="cones_3d",
                marker_id=int(tr.track_id),
                marker_type="CYLINDER",
                frame_id=frame_id,
                stamp_sec=stamp_sec,
                x=x,
                y=y,
                z=z,
                yaw_rad=0.0,
                scale_x=cone_d,
                scale_y=cone_d,
                scale_z=cone_h,
                rgba=(base_r, base_g, base_b, alpha),
            )
        )

        # 2. 2-Sigma Position Uncertainty Ellipse Marker
        cov = tr.covariance
        maj_d, min_d, ell_yaw = compute_covariance_ellipse_2sigma(
            cov[0][0], cov[0][1], cov[1][1]
        )
        markers.append(
            DebugMarkerDescriptor(
                ns="cones_cov_2sigma",
                marker_id=int(tr.track_id),
                marker_type="ELLIPSE",
                frame_id=frame_id,
                stamp_sec=stamp_sec,
                x=x,
                y=y,
                z=z - 0.5 * cone_h + 0.02,
                yaw_rad=ell_yaw,
                scale_x=max(0.08, maj_d),
                scale_y=max(0.08, min_d),
                scale_z=0.015,
                rgba=(base_r, base_g, base_b, 0.32),
            )
        )

        # 3. Track ID + Semantic Probability Label
        markers.append(
            DebugMarkerDescriptor(
                ns="cones_labels",
                marker_id=int(tr.track_id),
                marker_type="TEXT_VIEW_FACING",
                frame_id=frame_id,
                stamp_sec=stamp_sec,
                x=x,
                y=y,
                z=z + 0.5 * cone_h + 0.18,
                yaw_rad=0.0,
                scale_x=0.16,
                scale_y=0.16,
                scale_z=0.16,
                rgba=(1.0, 1.0, 1.0, 0.95),
                text=f"#{tr.track_id} {CLASS_NAMES[cls_idx]} ({max_p * 100.0:.0f}%)",
            )
        )

    if corridor is not None and corridor.centerline_waypoints:
        # Left boundary (Blue line strip)
        markers.append(
            DebugMarkerDescriptor(
                ns="corridor_boundaries",
                marker_id=1,
                marker_type="LINE_STRIP",
                frame_id=frame_id,
                stamp_sec=stamp_sec,
                x=0.0,
                y=0.0,
                z=0.0,
                yaw_rad=0.0,
                scale_x=0.06,
                scale_y=0.06,
                scale_z=0.06,
                rgba=(0.15, 0.52, 0.98, 0.85),
                points_xyz=[(px, py, -0.35) for px, py in corridor.left_boundary_xy],
            )
        )
        # Right boundary (Yellow line strip)
        markers.append(
            DebugMarkerDescriptor(
                ns="corridor_boundaries",
                marker_id=2,
                marker_type="LINE_STRIP",
                frame_id=frame_id,
                stamp_sec=stamp_sec,
                x=0.0,
                y=0.0,
                z=0.0,
                yaw_rad=0.0,
                scale_x=0.06,
                scale_y=0.06,
                scale_z=0.06,
                rgba=(0.98, 0.85, 0.12, 0.85),
                points_xyz=[(px, py, -0.35) for px, py in corridor.right_boundary_xy],
            )
        )
        # Centerline clothoid path (Emerald green line strip)
        markers.append(
            DebugMarkerDescriptor(
                ns="corridor_boundaries",
                marker_id=3,
                marker_type="LINE_STRIP",
                frame_id=frame_id,
                stamp_sec=stamp_sec,
                x=0.0,
                y=0.0,
                z=0.0,
                yaw_rad=0.0,
                scale_x=0.08,
                scale_y=0.08,
                scale_z=0.08,
                rgba=(0.12, 0.92, 0.45, 0.92),
                points_xyz=[
                    (wx, wy, -0.35) for wx, wy, _, _ in corridor.centerline_waypoints
                ],
            )
        )

    return markers


def build_reprojection_debug_annotations(
    clusters_xyz: Sequence[Tuple[float, float, float]],
    detections: Sequence[DetectionRecord],
    intrinsics: CameraIntrinsics,
    extrinsics: LidarToCameraExtrinsics,
    gate_px: float = 100.0,
) -> List[ReprojectionAnnotation]:
    """Project 3D cone frustum wireframes into the 2D camera frame and compute residual diagnostics."""
    annotations: List[ReprojectionAnnotation] = []
    half_r_base = 0.114
    half_r_top = 0.035
    half_h = 0.1625

    for idx, (cx, cy, cz) in enumerate(clusters_xyz):
        proj_c = project_lidar_point_to_image(
            (cx, cy, cz), intrinsics, extrinsics, min_depth_m=0.5, max_depth_m=60.0
        )
        if proj_c is None:
            continue
        pu, pv, pdepth = proj_c

        # Project 4 frustum wireframe corners (base left/right, top left/right)
        corners_3d = [
            (cx, cy + half_r_base, cz - half_h),
            (cx, cy - half_r_base, cz - half_h),
            (cx, cy - half_r_top, cz + half_h),
            (cx, cy + half_r_top, cz + half_h),
        ]
        poly_uv: List[Tuple[float, float]] = []
        for pt in corners_3d:
            p = project_lidar_point_to_image(
                pt, intrinsics, extrinsics, min_depth_m=0.5, max_depth_m=60.0
            )
            if p is not None:
                poly_uv.append((round(float(p[0]), 2), round(float(p[1]), 2)))

        best_j = -1
        best_res = float(gate_px)
        best_label = "UNMATCHED"
        for j, det in enumerate(detections):
            du = (
                0.5 * float(det.xmin + det.xmax)
                if hasattr(det, "xmin")
                else float(getattr(det, "u", 0.0))
            )
            dv = (
                0.5 * float(det.ymin + det.ymax)
                if hasattr(det, "ymin")
                else float(getattr(det, "v", 0.0))
            )
            d = math.hypot(pu - du, pv - dv)
            if d < best_res:
                best_res = d
                best_j = j
                probs = list(det.class_probs) if det.class_probs else [0.25] * 4
                c_idx = int(max(range(min(4, len(probs))), key=lambda i: probs[i]))
                best_label = CLASS_NAMES[c_idx]

        annotations.append(
            ReprojectionAnnotation(
                cluster_index=idx,
                u_proj=round(float(pu), 2),
                v_proj=round(float(pv), 2),
                depth_m=round(float(pdepth), 3),
                frustum_polygon_uv=poly_uv,
                matched_detection_index=best_j,
                residual_px=round(float(best_res if best_j >= 0 else -1.0), 2),
                class_label=best_label,
            )
        )

    return annotations
