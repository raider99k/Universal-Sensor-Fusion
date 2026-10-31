from __future__ import annotations

import ctypes
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
                _LOADED_LIB = lib
                return _LOADED_LIB
            except Exception:
                continue
    return None
