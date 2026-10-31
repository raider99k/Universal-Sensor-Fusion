from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from scipy.optimize import least_squares as _scipy_least_squares
except Exception:  # pragma: no cover
    _scipy_least_squares = None  # type: ignore[assignment]

from fusion_core.early_fusion import (
    CameraIntrinsics,
    DetectionRecord,
    LidarToCameraExtrinsics,
    project_lidar_point_to_image,
)
from fusion_core.ekf_tracker import EgoMotionStep, transform_point_between_frames


@dataclass
class GroundPlaneCalibrationResult:
    ground_z_m: float
    resting_pitch_rad: float
    resting_roll_rad: float
    recommended_ground_z_max_m: float
    inlier_rmse_m: float
    inlier_count: int


@dataclass
class ExtrinsicCalibrationResult:
    initial_rmse_px: float
    final_rmse_px: float
    matched_pairs: int
    delta_roll_rad: float
    delta_pitch_rad: float
    delta_yaw_rad: float
    delta_tx_m: float
    delta_ty_m: float
    delta_tz_m: float
    calibrated_extrinsics: LidarToCameraExtrinsics


@dataclass
class TemporalSyncCalibrationResult:
    estimated_time_offset_sec: float
    zero_offset_rmse_px: float
    optimal_offset_rmse_px: float
    evaluated_frames: int


@dataclass
class CalibrationFrameSample:
    """Single synchronized or near-synchronized calibration sample."""
    clusters_xyz: List[Tuple[float, float, float]]
    detections: List[DetectionRecord]
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    yaw_rate_radps: float = 0.0


def calibrate_lidar_ground_plane(
    points_xyz: Sequence[Tuple[float, float, float]],
    x_min_m: float = 3.0,
    x_max_m: float = 18.0,
    y_abs_max_m: float = 4.5,
    z_max_seed_m: float = -0.05,
    cell_size_m: float = 1.0,
    irls_iterations: int = 6,
    huber_delta_m: float = 0.035,
) -> GroundPlaneCalibrationResult:
    """Estimate resting LiDAR ground height, pitch, and roll from raw 3D road points.

    Uses lowest-quantile per-grid-cell seeding followed by Iteratively Reweighted
    Least Squares (IRLS) with a Huber loss so cone points and trackside curbs do not
    bias the asphalt plane fit:
        z_asphalt(x, y) = a * x + b * y + z0
    """
    cell_min_z: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for x, y, z in points_xyz:
        if x < x_min_m or x > x_max_m or abs(y) > y_abs_max_m or z > z_max_seed_m:
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        key = (int(math.floor(x / cell_size_m)), int(math.floor(y / cell_size_m)))
        prev = cell_min_z.get(key)
        if prev is None or z < prev[2]:
            cell_min_z[key] = (float(x), float(y), float(z))

    seeds = list(cell_min_z.values())
    if len(seeds) < 4:
        # Fallback if very few points are available
        zs = [float(p[2]) for p in points_xyz if math.isfinite(p[2]) and p[2] <= z_max_seed_m]
        z0 = sorted(zs)[len(zs) // 4] if zs else -0.45
        return GroundPlaneCalibrationResult(
            ground_z_m=float(z0),
            resting_pitch_rad=0.0,
            resting_roll_rad=0.0,
            recommended_ground_z_max_m=float(z0 + 0.085),
            inlier_rmse_m=0.0,
            inlier_count=len(zs),
        )

    # Solve weighted plane z = a*x + b*y + z0 via normal equations
    a, b, z0 = 0.0, 0.0, -0.45
    weights = [1.0] * len(seeds)

    for _ in range(max(1, irls_iterations)):
        s_xx = s_xy = s_x1 = s_yy = s_y1 = s_11 = 0.0
        s_xz = s_yz = s_1z = 0.0
        for (x, y, z), w in zip(seeds, weights):
            s_xx += w * x * x
            s_xy += w * x * y
            s_x1 += w * x
            s_yy += w * y * y
            s_y1 += w * y
            s_11 += w
            s_xz += w * x * z
            s_yz += w * y * z
            s_1z += w * z

        if np is not None:
            mat = np.array(
                [[s_xx + 1e-6, s_xy, s_x1], [s_xy, s_yy + 1e-6, s_y1], [s_x1, s_y1, s_11 + 1e-6]],
                dtype=np.float64,
            )
            rhs = np.array([s_xz, s_yz, s_1z], dtype=np.float64)
            try:
                sol = np.linalg.solve(mat, rhs)
                a, b, z0 = float(sol[0]), float(sol[1]), float(sol[2])
            except Exception:
                break
        else:
            z0 = s_1z / max(1e-6, s_11)

        new_weights: List[float] = []
        for x, y, z in seeds:
            res = abs(z - (a * x + b * y + z0))
            if res <= huber_delta_m:
                new_weights.append(1.0)
            else:
                new_weights.append(huber_delta_m / max(res, 1e-6))
        weights = new_weights

    inlier_sq_errs: List[float] = []
    for x, y, z in seeds:
        res = z - (a * x + b * y + z0)
        if abs(res) <= 2.5 * huber_delta_m:
            inlier_sq_errs.append(res * res)

    inlier_rmse = math.sqrt(sum(inlier_sq_errs) / max(1, len(inlier_sq_errs)))
    norm_n = math.sqrt(a * a + b * b + 1.0)
    # Since z_leveled = z + x * sin(pitch) - y * sin(roll) == z0 when z = a*x + b*y + z0:
    # sin(pitch) = -a / norm_n, sin(roll) = b / norm_n
    resting_pitch = math.asin(max(-0.5, min(0.5, -a / norm_n)))
    resting_roll = math.asin(max(-0.5, min(0.5, b / norm_n)))

    return GroundPlaneCalibrationResult(
        ground_z_m=round(float(z0), 5),
        resting_pitch_rad=round(float(resting_pitch), 6),
        resting_roll_rad=round(float(resting_roll), 6),
        recommended_ground_z_max_m=round(float(z0 + 0.085), 5),
        inlier_rmse_m=round(float(inlier_rmse), 5),
        inlier_count=len(inlier_sq_errs),
    )


def _compose_extrinsics_with_delta(
    base_ext: LidarToCameraExtrinsics,
    d_roll: float,
    d_pitch: float,
    d_yaw: float,
    d_tx: float,
    d_ty: float,
    d_tz: float,
) -> LidarToCameraExtrinsics:
    """Apply a small camera-frame 6-DoF perturbation (d_roll, d_pitch, d_yaw, d_t) to base extrinsics."""
    delta_ext = LidarToCameraExtrinsics.from_rpy_translation(
        tx=0.0, ty=0.0, tz=0.0, roll=d_roll, pitch=d_pitch, yaw=d_yaw
    )
    dr = delta_ext.rotation
    br = base_ext.rotation
    # R_new = dR * R_base
    r_new = [
        [
            dr[0][0] * br[0][0] + dr[0][1] * br[1][0] + dr[0][2] * br[2][0],
            dr[0][0] * br[0][1] + dr[0][1] * br[1][1] + dr[0][2] * br[2][1],
            dr[0][0] * br[0][2] + dr[0][1] * br[1][2] + dr[0][2] * br[2][2],
        ],
        [
            dr[1][0] * br[0][0] + dr[1][1] * br[1][0] + dr[1][2] * br[2][0],
            dr[1][0] * br[0][1] + dr[1][1] * br[1][1] + dr[1][2] * br[2][1],
            dr[1][0] * br[0][2] + dr[1][1] * br[1][2] + dr[1][2] * br[2][2],
        ],
        [
            dr[2][0] * br[0][0] + dr[2][1] * br[1][0] + dr[2][2] * br[2][0],
            dr[2][0] * br[0][1] + dr[2][1] * br[1][1] + dr[2][2] * br[2][1],
            dr[2][0] * br[0][2] + dr[2][1] * br[1][2] + dr[2][2] * br[2][2],
        ],
    ]
    t_new = (
        base_ext.translation[0] + d_tx,
        base_ext.translation[1] + d_ty,
        base_ext.translation[2] + d_tz,
    )
    return LidarToCameraExtrinsics(rotation=r_new, translation=t_new)


def _det_center_uv(det: DetectionRecord) -> Tuple[float, float]:
    if hasattr(det, "xmin") and hasattr(det, "xmax"):
        return (0.5 * float(det.xmin + det.xmax), 0.5 * float(det.ymin + det.ymax))
    return (float(getattr(det, "u", 0.0)), float(getattr(det, "v", 0.0)))


def calibrate_lidar_to_camera_extrinsics(
    frames: Sequence[CalibrationFrameSample],
    intrinsics: CameraIntrinsics,
    initial_extrinsics: LidarToCameraExtrinsics,
    association_gate_px: float = 140.0,
    optimize_translation: bool = True,
) -> ExtrinsicCalibrationResult:
    """Automatically refine 6-DoF LiDAR-to-Camera extrinsics from 3D clusters and 2D boxes.

    Matches 3D LiDAR cone clusters to 2D camera detections within `association_gate_px`
    and minimizes Huber-weighted 2D reprojection residuals across two ICP-style passes.
    """
    def _associate(ext_active: LidarToCameraExtrinsics) -> List[Tuple[Tuple[float, float, float], Tuple[float, float]]]:
        pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float]]] = []
        for frame in frames:
            used_det = set()
            for cx, cy, cz in frame.clusters_xyz:
                proj = project_lidar_point_to_image(
                    (cx, cy, cz), intrinsics, ext_active, min_depth_m=0.5, max_depth_m=60.0
                )
                if proj is None:
                    continue
                pu, pv, _ = proj
                best_d = association_gate_px
                best_j = -1
                for j, det in enumerate(frame.detections):
                    if j in used_det:
                        continue
                    du, dv = _det_center_uv(det)
                    d = math.hypot(pu - du, pv - dv)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    used_det.add(best_j)
                    du, dv = _det_center_uv(frame.detections[best_j])
                    pairs.append(((float(cx), float(cy), float(cz)), (du, dv)))
        return pairs

    correspondences = _associate(initial_extrinsics)

    if not correspondences:
        return ExtrinsicCalibrationResult(
            initial_rmse_px=0.0,
            final_rmse_px=0.0,
            matched_pairs=0,
            delta_roll_rad=0.0,
            delta_pitch_rad=0.0,
            delta_yaw_rad=0.0,
            delta_tx_m=0.0,
            delta_ty_m=0.0,
            delta_tz_m=0.0,
            calibrated_extrinsics=initial_extrinsics,
        )

    def _compute_residuals(params: Sequence[float]) -> List[float]:
        d_roll, d_pitch, d_yaw = float(params[0]), float(params[1]), float(params[2])
        if optimize_translation and len(params) >= 6:
            d_tx, d_ty, d_tz = float(params[3]), float(params[4]), float(params[5])
        else:
            d_tx = d_ty = d_tz = 0.0
        ext = _compose_extrinsics_with_delta(
            initial_extrinsics, d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz
        )
        res: List[float] = []
        for pt3d, (u_obs, v_obs) in correspondences:
            proj = project_lidar_point_to_image(
                pt3d, intrinsics, ext, min_depth_m=0.5, max_depth_m=60.0
            )
            if proj is None:
                res.extend([200.0, 200.0])
            else:
                res.extend([proj[0] - u_obs, proj[1] - v_obs])
        # Mild Tikhonov prior on translation so collinear cone geometry stays well-conditioned
        if optimize_translation and len(params) >= 6:
            res.extend([d_tx * 15.0, d_ty * 15.0, d_tz * 15.0])
        return res

    init_res = _compute_residuals([0.0] * (6 if optimize_translation else 3))
    n_pix = 2 * len(correspondences)
    initial_rmse = math.sqrt(sum(r * r for r in init_res[:n_pix]) / max(1, n_pix // 2))

    best_params = [0.0] * (6 if optimize_translation else 3)
    if _scipy_least_squares is not None and np is not None:
        x0 = np.zeros(6 if optimize_translation else 3, dtype=np.float64)
        bounds_low = [-0.25, -0.25, -0.25] + ([-0.35, -0.35, -0.35] if optimize_translation else [])
        bounds_high = [0.25, 0.25, 0.25] + ([0.35, 0.35, 0.35] if optimize_translation else [])
        for _pass in range(2):
            sol = _scipy_least_squares(
                lambda x: np.asarray(_compute_residuals(x), dtype=np.float64),
                x0,
                bounds=(bounds_low, bounds_high),
                loss="huber",
                f_scale=4.0,
                max_nfev=80,
            )
            best_params = [float(v) for v in sol.x]
            x0 = np.asarray(best_params, dtype=np.float64)
            ext_pass = _compose_extrinsics_with_delta(
                initial_extrinsics,
                best_params[0],
                best_params[1],
                best_params[2],
                best_params[3] if optimize_translation else 0.0,
                best_params[4] if optimize_translation else 0.0,
                best_params[5] if optimize_translation else 0.0,
            )
            new_corr = _associate(ext_pass)
            if len(new_corr) >= len(correspondences):
                correspondences = new_corr
                n_pix = 2 * len(correspondences)

    d_roll = best_params[0]
    d_pitch = best_params[1]
    d_yaw = best_params[2]
    d_tx = best_params[3] if optimize_translation else 0.0
    d_ty = best_params[4] if optimize_translation else 0.0
    d_tz = best_params[5] if optimize_translation else 0.0

    final_res = _compute_residuals(best_params)
    final_rmse = math.sqrt(sum(r * r for r in final_res[:n_pix]) / max(1, n_pix // 2))
    calibrated_ext = _compose_extrinsics_with_delta(
        initial_extrinsics, d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz
    )

    return ExtrinsicCalibrationResult(
        initial_rmse_px=round(float(initial_rmse), 3),
        final_rmse_px=round(float(final_rmse), 3),
        matched_pairs=len(correspondences),
        delta_roll_rad=round(float(d_roll), 6),
        delta_pitch_rad=round(float(d_pitch), 6),
        delta_yaw_rad=round(float(d_yaw), 6),
        delta_tx_m=round(float(d_tx), 5),
        delta_ty_m=round(float(d_ty), 5),
        delta_tz_m=round(float(d_tz), 5),
        calibrated_extrinsics=calibrated_ext,
    )


def calibrate_temporal_sync_offset(
    frames: Sequence[CalibrationFrameSample],
    intrinsics: CameraIntrinsics,
    extrinsics: LidarToCameraExtrinsics,
    search_range_sec: Tuple[float, float] = (-0.08, 0.08),
    association_gate_px: float = 140.0,
) -> TemporalSyncCalibrationResult:
    """Estimate Camera-to-LiDAR clock offset delta_t (sec) from dynamic yaw maneuvers.

    Evaluates 2D reprojection RMSE as a function of time shift dt where 3D LiDAR
    points are propagated across SE(2) ego-motion over dt seconds.
    """
    dynamic_frames = [f for f in frames if abs(f.yaw_rate_radps) >= 0.08 or abs(f.vx_mps) >= 1.5]
    eval_frames = dynamic_frames if dynamic_frames else list(frames)
    if not eval_frames:
        return TemporalSyncCalibrationResult(
            estimated_time_offset_sec=0.0,
            zero_offset_rmse_px=0.0,
            optimal_offset_rmse_px=0.0,
            evaluated_frames=0,
        )

    def _rmse_at_dt(dt_sec: float) -> float:
        sq_errs: List[float] = []
        for frame in eval_frames:
            step = EgoMotionStep(
                dt=abs(dt_sec),
                vx=frame.vx_mps if dt_sec >= 0.0 else -frame.vx_mps,
                vy=frame.vy_mps if dt_sec >= 0.0 else -frame.vy_mps,
                wz=frame.yaw_rate_radps if dt_sec >= 0.0 else -frame.yaw_rate_radps,
            )
            for cx, cy, cz in frame.clusters_xyz:
                if abs(dt_sec) > 1e-6:
                    tx, ty = transform_point_between_frames(cx, cy, [step])
                else:
                    tx, ty = cx, cy
                proj = project_lidar_point_to_image(
                    (tx, ty, cz), intrinsics, extrinsics, min_depth_m=0.5, max_depth_m=60.0
                )
                if proj is None:
                    continue
                pu, pv, _ = proj
                best_d = association_gate_px
                for det in frame.detections:
                    du, dv = _det_center_uv(det)
                    d = math.hypot(pu - du, pv - dv)
                    if d < best_d:
                        best_d = d
                if best_d < association_gate_px:
                    sq_errs.append(best_d * best_d)
        if not sq_errs:
            return float(association_gate_px)
        return math.sqrt(sum(sq_errs) / len(sq_errs))

    zero_rmse = _rmse_at_dt(0.0)
    t_min, t_max = float(search_range_sec[0]), float(search_range_sec[1])
    # Coarse grid search (33 steps) followed by golden-section refinement
    n_grid = 33
    best_dt = 0.0
    best_rmse = zero_rmse
    for i in range(n_grid):
        dt = t_min + (t_max - t_min) * (i / max(1, n_grid - 1))
        r = _rmse_at_dt(dt)
        if r < best_rmse - 1e-4:
            best_rmse = r
            best_dt = dt

    step_w = (t_max - t_min) / max(1, n_grid - 1)
    a = max(t_min, best_dt - step_w)
    b = min(t_max, best_dt + step_w)
    gr = (math.sqrt(5.0) - 1.0) * 0.5
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _rmse_at_dt(c)
    fd = _rmse_at_dt(d)
    for _ in range(10):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _rmse_at_dt(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _rmse_at_dt(d)

    opt_dt = 0.5 * (a + b)
    opt_rmse = _rmse_at_dt(opt_dt)
    if opt_rmse >= zero_rmse - 1e-3:
        opt_dt = 0.0
        opt_rmse = zero_rmse

    return TemporalSyncCalibrationResult(
        estimated_time_offset_sec=round(float(opt_dt), 5),
        zero_offset_rmse_px=round(float(zero_rmse), 3),
        optimal_offset_rmse_px=round(float(opt_rmse), 3),
        evaluated_frames=len(eval_frames),
    )


def format_calibrated_ros2_yaml(
    ground_result: Optional[GroundPlaneCalibrationResult] = None,
    extrinsic_result: Optional[ExtrinsicCalibrationResult] = None,
    sync_result: Optional[TemporalSyncCalibrationResult] = None,
) -> str:
    """Format auto-calibrated parameters as a ready-to-paste ROS 2 YAML configuration block."""
    lines = ["# Auto-generated by fusion_core.auto_calibrate"]
    if ground_result is not None:
        lines.extend(
            [
                "ouster_lidar_adapter:",
                "  ros__parameters:",
                f"    ground_z_max: {ground_result.recommended_ground_z_max_m:.4f}",
                f"    resting_pitch_rad: {ground_result.resting_pitch_rad:.5f}",
                f"    resting_roll_rad: {ground_result.resting_roll_rad:.5f}",
            ]
        )
    if extrinsic_result is not None or sync_result is not None:
        lines.extend(
            [
                "fusion_node:",
                "  ros__parameters:",
            ]
        )
        if extrinsic_result is not None:
            t = extrinsic_result.calibrated_extrinsics.translation
            lines.extend(
                [
                    f"    delta_extrinsic_roll_rad: {extrinsic_result.delta_roll_rad:.5f}",
                    f"    delta_extrinsic_pitch_rad: {extrinsic_result.delta_pitch_rad:.5f}",
                    f"    delta_extrinsic_yaw_rad: {extrinsic_result.delta_yaw_rad:.5f}",
                    f"    extrinsic_tx: {t[0]:.4f}",
                    f"    extrinsic_ty: {t[1]:.4f}",
                    f"    extrinsic_tz: {t[2]:.4f}",
                ]
            )
        if sync_result is not None:
            lines.append(f"    camera_time_offset_sec: {sync_result.estimated_time_offset_sec:.5f}")
    return "\n".join(lines) + "\n"
