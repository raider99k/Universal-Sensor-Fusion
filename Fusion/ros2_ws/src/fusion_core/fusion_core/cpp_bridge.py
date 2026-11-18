from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


class LidarPipelineConfigC(ctypes.Structure):
    _fields_ = [
        ("n_total", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("point_step", ctypes.c_int32),
        ("off_x", ctypes.c_int32),
        ("off_y", ctypes.c_int32),
        ("off_z", ctypes.c_int32),
        ("off_t", ctypes.c_int32),
        ("t_datatype", ctypes.c_int32),
        ("off_stripe", ctypes.c_int32),
        ("stripe_datatype", ctypes.c_int32),
        ("sensor_yaw_rad", ctypes.c_double),
        ("sensor_offset_x_m", ctypes.c_double),
        ("sensor_offset_y_m", ctypes.c_double),
        ("sensor_offset_z_m", ctypes.c_double),
        ("enable_imu_leveling", ctypes.c_int32),
        ("sin_pitch", ctypes.c_double),
        ("sin_roll", ctypes.c_double),
        ("roi_x_min", ctypes.c_double),
        ("roi_x_max", ctypes.c_double),
        ("roi_y_min", ctypes.c_double),
        ("roi_y_max", ctypes.c_double),
        ("roi_z_min", ctypes.c_double),
        ("roi_z_max", ctypes.c_double),
        ("enable_self_return_suppression", ctypes.c_int32),
        ("self_return_radius_m", ctypes.c_double),
        ("self_return_z_min_m", ctypes.c_double),
        ("self_return_z_max_m", ctypes.c_double),
        ("self_return_min_range_m", ctypes.c_double),
        ("max_points_for_clustering", ctypes.c_int32),
        ("has_motion", ctypes.c_int32),
        ("vx", ctypes.c_double),
        ("vy", ctypes.c_double),
        ("wz", ctypes.c_double),
        ("deskew_scan_duration_s", ctypes.c_double),
        ("deskew_max_dt_s", ctypes.c_double),
        ("deskew_spin_direction", ctypes.c_double),
        ("enable_adaptive_ground_model", ctypes.c_int32),
        ("ground_z_max", ctypes.c_double),
        ("ground_zone_radial_step_m", ctypes.c_double),
        ("ground_zone_azimuth_bins", ctypes.c_int32),
        ("max_ground_slope_rad", ctypes.c_double),
        ("ground_clearance_m", ctypes.c_double),
        ("cone_max_radius_m", ctypes.c_double),
        ("enable_ground_point_recovery", ctypes.c_int32),
        ("adaptive_range_ref_m", ctypes.c_double),
        ("cluster_tolerance_m", ctypes.c_double),
    ]


class LidarExtractStatsC(ctypes.Structure):
    _fields_ = [
        ("raw_points", ctypes.c_int32),
        ("kept_points", ctypes.c_int32),
        ("self_return_rejects", ctypes.c_int32),
    ]


class ClusterBuildConfigC(ctypes.Structure):
    _fields_ = [
        ("cluster_tolerance_m", ctypes.c_double),
        ("min_cluster_points", ctypes.c_int32),
        ("max_cluster_points", ctypes.c_int32),
        ("enable_adaptive_clustering", ctypes.c_int32),
        ("min_cluster_points_far", ctypes.c_int32),
        ("adaptive_range_ref_m", ctypes.c_double),
        ("cone_min_height_m", ctypes.c_double),
        ("cone_max_height_m", ctypes.c_double),
        ("cone_max_radius_m", ctypes.c_double),
        ("covariance_floor_xy", ctypes.c_double),
        ("covariance_floor_z", ctypes.c_double),
        ("enable_surface_axis_offset", ctypes.c_int32),
        ("surface_axis_offset_m", ctypes.c_double),
        ("enable_stripe_classification", ctypes.c_int32),
        ("has_refl", ctypes.c_int32),
    ]


class ClusterDescriptorC(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("cz", ctypes.c_double),
        ("var_x", ctypes.c_double),
        ("cov_xy", ctypes.c_double),
        ("var_y", ctypes.c_double),
        ("var_z", ctypes.c_double),
        ("geom_quality", ctypes.c_double),
        ("stripe_ratio", ctypes.c_double),
        ("point_count", ctypes.c_int32),
    ]


class CameraIntrinsicsC(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_double),
        ("fy", ctypes.c_double),
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("k1", ctypes.c_double),
        ("k2", ctypes.c_double),
        ("p1", ctypes.c_double),
        ("p2", ctypes.c_double),
        ("k3", ctypes.c_double),
    ]


class ExtrinsicsC(ctypes.Structure):
    _fields_ = [
        ("r", ctypes.c_double * 9),
        ("t", ctypes.c_double * 3),
        ("min_depth_m", ctypes.c_double),
        ("max_depth_m", ctypes.c_double),
        ("sample_radius_scale", ctypes.c_double),
        ("min_semantic_weight", ctypes.c_double),
        ("no_match_semantic_weight", ctypes.c_double),
        ("soft_likelihood_enabled", ctypes.c_int32),
        ("soft_likelihood_sigma_scale", ctypes.c_double),
        ("soft_likelihood_min_value", ctypes.c_double),
        ("soft_likelihood_outside_ratio", ctypes.c_double),
        ("enable_frustum_hull_sampling", ctypes.c_int32),
        ("enable_triangular_cone_mask", ctypes.c_int32),
        ("cone_mask_apex_width_ratio", ctypes.c_double),
    ]


class ClusterInputC(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("cz", ctypes.c_double),
        ("cov_xx", ctypes.c_double),
        ("cov_yy", ctypes.c_double),
        ("cov_zz", ctypes.c_double),
    ]


class Detection2DC(ctypes.Structure):
    _fields_ = [
        ("xmin", ctypes.c_double),
        ("ymin", ctypes.c_double),
        ("xmax", ctypes.c_double),
        ("ymax", ctypes.c_double),
        ("class_probs", ctypes.c_double * 4),
        ("confidence", ctypes.c_double),
        ("polygon_count", ctypes.c_int32),
        ("polygon_u", ctypes.c_double * 16),
        ("polygon_v", ctypes.c_double * 16),
        ("is_big_orange", ctypes.c_int32),
    ]


class StereoDepthImageC(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("encoding", ctypes.c_int32),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("row_step_bytes", ctypes.c_int32),
        ("depth_scale", ctypes.c_double),
        ("baseline_m", ctypes.c_double),
        ("disparity_sigma_px", ctypes.c_double),
        ("min_depth_m", ctypes.c_double),
        ("max_depth_m", ctypes.c_double),
    ]


class FrustumRecoveryConfigC(ctypes.Structure):
    _fields_ = [
        ("cone_height_m", ctypes.c_double),
        ("big_orange_height_m", ctypes.c_double),
        ("min_confidence", ctypes.c_double),
        ("min_bbox_height_px", ctypes.c_double),
        ("min_depth_m", ctypes.c_double),
        ("max_depth_m", ctypes.c_double),
        ("dedup_radius_m", ctypes.c_double),
        ("range_std_fraction", ctypes.c_double),
        ("bearing_std_fraction", ctypes.c_double),
        ("enable_ground_plane_depth_fusion", ctypes.c_int32),
        ("ground_z_base_m", ctypes.c_double),
        ("ground_slope_rad", ctypes.c_double),
        ("enable_stereo_depth_fusion", ctypes.c_int32),
        ("enable_triangular_cone_mask", ctypes.c_int32),
        ("cone_mask_apex_width_ratio", ctypes.c_double),
    ]


class RecoveredConeC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("covariance", ctypes.c_double * 9),
        ("class_probs", ctypes.c_double * 4),
        ("confidence", ctypes.c_double),
        ("depth_m", ctypes.c_double),
        ("detection_idx", ctypes.c_int32),
        ("used_stereo", ctypes.c_int32),
    ]


class ClusterSemanticC(ctypes.Structure):
    _fields_ = [
        ("class_probs", ctypes.c_double * 4),
        ("semantic_weight", ctypes.c_double),
        ("projected_samples", ctypes.c_int32),
        ("matched_samples", ctypes.c_int32),
        ("matched_weight_sum", ctypes.c_double),
        ("best_confidence", ctypes.c_double),
        ("residual_samples", ctypes.c_int32),
        ("residual_sum_px", ctypes.c_double),
        ("residual_max_px", ctypes.c_double),
        ("matched_detection_idx", ctypes.c_int32),
    ]


class TrackerConfigC(ctypes.Structure):
    _fields_ = [
        ("mahalanobis_gate", ctypes.c_double),
        ("semantic_min_overlap", ctypes.c_double),
        ("semantic_cost_weight", ctypes.c_double),
        ("process_noise_accel", ctypes.c_double),
        ("min_meas_var", ctypes.c_double),
        ("confirm_hits", ctypes.c_int32),
        ("max_missed", ctypes.c_int32),
        ("class_update_alpha", ctypes.c_double),
        ("max_predict_dt", ctypes.c_double),
        ("motion_model_landmark", ctypes.c_int32),
        ("assignment_hungarian", ctypes.c_int32),
        ("semantic_dirichlet", ctypes.c_int32),
        ("dirichlet_prior_strength", ctypes.c_double),
        ("dirichlet_evidence_scale", ctypes.c_double),
        ("dirichlet_max_concentration", ctypes.c_double),
        ("landmark_process_noise_pos", ctypes.c_double),
        ("odom_vx_std", ctypes.c_double),
        ("odom_vy_std", ctypes.c_double),
        ("odom_wz_std", ctypes.c_double),
        ("corridor_prior_weight", ctypes.c_double),
        ("enable_global_map", ctypes.c_int32),
        ("global_map_gate_radius_m", ctypes.c_double),
        ("global_map_min_lap_distance_m", ctypes.c_double),
    ]


class CorridorEstimateC(ctypes.Structure):
    _fields_ = [
        ("xm", ctypes.c_double),
        ("y0", ctypes.c_double),
        ("slope", ctypes.c_double),
        ("curvature_kappa", ctypes.c_double),
        ("track_width_m", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("n_waypoints", ctypes.c_int32),
        ("left_x", ctypes.c_double * 64),
        ("left_y", ctypes.c_double * 64),
        ("right_x", ctypes.c_double * 64),
        ("right_y", ctypes.c_double * 64),
        ("wp_x", ctypes.c_double * 64),
        ("wp_y", ctypes.c_double * 64),
        ("wp_heading", ctypes.c_double * 64),
        ("wp_kappa", ctypes.c_double * 64),
    ]


class OnlineCalibPairC(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("cz", ctypes.c_double),
        ("det_u", ctypes.c_double),
        ("det_v", ctypes.c_double),
        ("det_w", ctypes.c_double),
        ("det_h", ctypes.c_double),
    ]


class OnlineCalibResultC(ctypes.Structure):
    _fields_ = [
        ("matched_pairs", ctypes.c_int32),
        ("initial_rmse_px", ctypes.c_double),
        ("final_rmse_px", ctypes.c_double),
        ("delta_roll_rad", ctypes.c_double),
        ("delta_pitch_rad", ctypes.c_double),
        ("delta_yaw_rad", ctypes.c_double),
        ("refined_r", ctypes.c_double * 9),
        ("refined_t", ctypes.c_double * 3),
    ]


class Observation2DC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("cov_xx", ctypes.c_double),
        ("cov_xy", ctypes.c_double),
        ("cov_yy", ctypes.c_double),
        ("z_var", ctypes.c_double),
        ("class_probs", ctypes.c_double * 4),
        ("quality", ctypes.c_double),
    ]


class EgoMotionC(ctypes.Structure):
    _fields_ = [
        ("has_motion", ctypes.c_int32),
        ("vx", ctypes.c_double),
        ("vy", ctypes.c_double),
        ("wz", ctypes.c_double),
        ("dt", ctypes.c_double),
        ("has_disp", ctypes.c_int32),
        ("dx", ctypes.c_double),
        ("dy", ctypes.c_double),
        ("dyaw", ctypes.c_double),
    ]


class TrackEstimateC(ctypes.Structure):
    _fields_ = [
        ("track_id", ctypes.c_int32),
        ("state", ctypes.c_double * 4),
        ("covariance", ctypes.c_double * 16),
        ("z", ctypes.c_double),
        ("z_var", ctypes.c_double),
        ("class_probs", ctypes.c_double * 4),
        ("hits", ctypes.c_int32),
        ("misses", ctypes.c_int32),
        ("quality", ctypes.c_double),
        ("status", ctypes.c_int32),
        ("first_stamp", ctypes.c_double),
        ("last_stamp", ctypes.c_double),
        ("semantic_certainty", ctypes.c_double),
    ]


class TrackerStepStatsC(ctypes.Structure):
    _fields_ = [
        ("tracks_created", ctypes.c_int32),
        ("tracks_deleted", ctypes.c_int32),
        ("assoc_candidates", ctypes.c_int32),
        ("assoc_matches", ctypes.c_int32),
        ("assoc_rejects", ctypes.c_int32),
        ("n_lost_tracks", ctypes.c_int32),
    ]


_LOADED_LIB: Optional[ctypes.CDLL] = None
_TRIED_LOAD = False


def get_cpp_kernels() -> Optional[ctypes.CDLL]:
    """Load the compiled C++20 native kernel shared library if available."""
    global _LOADED_LIB, _TRIED_LOAD
    if _TRIED_LOAD:
        return _LOADED_LIB
    _TRIED_LOAD = True

    src_root = Path(__file__).resolve().parents[2]
    candidates = [
        src_root / "fusion_core_cpp" / "bin" / "fusion_fast_kernels.dll",
        src_root / "fusion_core_cpp" / "bin" / "libfusion_fast_kernels.so",
        src_root / "fusion_core_cpp" / "bin" / "libfusion_fast_kernels.dylib",
    ]
    # Also resolve installed ROS 2 colcon libraries (`install/fusion_core_cpp/lib` or `install/lib`).
    for env_var in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "LD_LIBRARY_PATH"):
        raw = os.environ.get(env_var, "")
        for entry in raw.split(os.pathsep):
            if not entry:
                continue
            p = Path(entry)
            candidates.extend(
                [
                    p / "lib" / "libfusion_fast_kernels.so",
                    p / "libfusion_fast_kernels.so",
                    p / "fusion_core_cpp" / "lib" / "libfusion_fast_kernels.so",
                ]
            )
    resolved_file = Path(__file__).resolve()
    for parent in resolved_file.parents:
        if (parent / "fusion_core_cpp" / "lib" / "libfusion_fast_kernels.so").is_file():
            candidates.append(parent / "fusion_core_cpp" / "lib" / "libfusion_fast_kernels.so")
            break
    for path in candidates:
        if path.is_file():
            try:
                lib = ctypes.CDLL(str(path))
                lib.cpp_extract_and_filter_ground.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(LidarPipelineConfigC),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_int32,
                    ctypes.POINTER(LidarExtractStatsC),
                ]
                lib.cpp_extract_and_filter_ground.restype = ctypes.c_int32

                lib.cpp_filter_ground_only.argtypes = [
                    ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(LidarPipelineConfigC),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_int32,
                ]
                lib.cpp_filter_ground_only.restype = ctypes.c_int32

                lib.cpp_cluster_and_build.argtypes = [
                    ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ClusterBuildConfigC),
                    ctypes.POINTER(ClusterDescriptorC),
                    ctypes.c_int32,
                ]
                lib.cpp_cluster_and_build.restype = ctypes.c_int32

                lib.cpp_estimate_batch_semantics.argtypes = [
                    ctypes.c_int32,
                    ctypes.POINTER(ClusterInputC),
                    ctypes.c_int32,
                    ctypes.POINTER(Detection2DC),
                    ctypes.POINTER(CameraIntrinsicsC),
                    ctypes.POINTER(ExtrinsicsC),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ClusterSemanticC),
                ]
                lib.cpp_estimate_batch_semantics.restype = ctypes.c_int32

                lib.cpp_tracker_create.argtypes = [ctypes.POINTER(TrackerConfigC)]
                lib.cpp_tracker_create.restype = ctypes.c_void_p

                lib.cpp_tracker_destroy.argtypes = [ctypes.c_void_p]
                lib.cpp_tracker_destroy.restype = None

                lib.cpp_tracker_reset.argtypes = [ctypes.c_void_p]
                lib.cpp_tracker_reset.restype = None

                lib.cpp_tracker_step.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_double,
                    ctypes.POINTER(EgoMotionC),
                    ctypes.c_int32,
                    ctypes.POINTER(Observation2DC),
                    ctypes.POINTER(TrackEstimateC),
                    ctypes.c_int32,
                    ctypes.POINTER(TrackEstimateC),
                    ctypes.c_int32,
                    ctypes.POINTER(TrackerStepStatsC),
                ]
                lib.cpp_tracker_step.restype = ctypes.c_int32

                lib.cpp_tracker_get_tracks.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(TrackEstimateC),
                    ctypes.c_int32,
                ]
                lib.cpp_tracker_get_tracks.restype = ctypes.c_int32

                lib.cpp_recover_unmatched_detections.argtypes = [
                    ctypes.c_int32,
                    ctypes.POINTER(Detection2DC),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int32,
                    ctypes.POINTER(ClusterInputC),
                    ctypes.POINTER(CameraIntrinsicsC),
                    ctypes.POINTER(ExtrinsicsC),
                    ctypes.POINTER(FrustumRecoveryConfigC),
                    ctypes.POINTER(StereoDepthImageC),
                    ctypes.POINTER(RecoveredConeC),
                    ctypes.c_int32,
                ]
                lib.cpp_recover_unmatched_detections.restype = ctypes.c_int32

                lib.cpp_tracker_snapshot_create.argtypes = [ctypes.c_void_p]
                lib.cpp_tracker_snapshot_create.restype = ctypes.c_void_p

                lib.cpp_tracker_snapshot_restore.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                lib.cpp_tracker_snapshot_restore.restype = None

                lib.cpp_tracker_snapshot_destroy.argtypes = [ctypes.c_void_p]
                lib.cpp_tracker_snapshot_destroy.restype = None

                lib.cpp_tracker_get_last_stamp.argtypes = [ctypes.c_void_p]
                lib.cpp_tracker_get_last_stamp.restype = ctypes.c_double

                lib.cpp_tracker_get_corridor.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(EgoMotionC),
                    ctypes.POINTER(CorridorEstimateC),
                ]
                lib.cpp_tracker_get_corridor.restype = ctypes.c_int32

                lib.cpp_online_calibrate_extrinsics.argtypes = [
                    ctypes.c_int32,
                    ctypes.POINTER(OnlineCalibPairC),
                    ctypes.POINTER(CameraIntrinsicsC),
                    ctypes.POINTER(ExtrinsicsC),
                    ctypes.c_int32,
                    ctypes.POINTER(OnlineCalibResultC),
                ]
                lib.cpp_online_calibrate_extrinsics.restype = ctypes.c_int32

                _LOADED_LIB = lib
                return _LOADED_LIB
            except Exception:
                continue
    return None
