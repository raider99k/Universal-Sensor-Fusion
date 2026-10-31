#pragma once

#include <cstdint>

#if defined(_WIN32) || defined(_WIN64)
#define FUSION_CPP_EXPORT extern "C" __declspec(dllexport)
#else
#define FUSION_CPP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace fusion_core_cpp {

struct LidarPipelineConfigC {
    int32_t n_total;
    int32_t height;
    int32_t width;
    int32_t point_step;
    int32_t off_x;
    int32_t off_y;
    int32_t off_z;
    int32_t off_t;            // -1 if absent
    int32_t t_datatype;       // PointField datatype (6=UINT32, 7=FLOAT32, 8=FLOAT64)
    int32_t off_stripe;       // -1 if absent
    int32_t stripe_datatype;  // PointField datatype (2=UINT8, 4=UINT16, 6=UINT32, 7=FLOAT32, 8=FLOAT64)

    double sensor_yaw_rad;
    double sensor_offset_x_m;
    double sensor_offset_y_m;
    double sensor_offset_z_m;

    int32_t enable_imu_leveling;
    double sin_pitch;
    double sin_roll;

    double roi_x_min;
    double roi_x_max;
    double roi_y_min;
    double roi_y_max;
    double roi_z_min;
    double roi_z_max;

    int32_t enable_self_return_suppression;
    double self_return_radius_m;
    double self_return_z_min_m;
    double self_return_z_max_m;
    double self_return_min_range_m;

    int32_t max_points_for_clustering;

    int32_t has_motion;  // 0: off, 1: normalized [0,1] backward phase, 2: raw nanosecond forward offset
    double vx;
    double vy;
    double wz;
    double deskew_scan_duration_s;
    double deskew_max_dt_s;
    double deskew_spin_direction;

    int32_t enable_adaptive_ground_model;
    double ground_z_max;
    double ground_zone_radial_step_m;
    int32_t ground_zone_azimuth_bins;
    double max_ground_slope_rad;
    double ground_clearance_m;
    double cone_max_radius_m;
    int32_t enable_ground_point_recovery;
    double adaptive_range_ref_m;
    double cluster_tolerance_m;
};

struct LidarExtractStatsC {
    int32_t raw_points;
    int32_t kept_points;
    int32_t self_return_rejects;
};

struct ClusterBuildConfigC {
    double cluster_tolerance_m;
    int32_t min_cluster_points;
    int32_t max_cluster_points;
    int32_t enable_adaptive_clustering;
    int32_t min_cluster_points_far;
    double adaptive_range_ref_m;
    double cone_min_height_m;
    double cone_max_height_m;
    double cone_max_radius_m;
    double covariance_floor_xy;
    double covariance_floor_z;
    int32_t enable_surface_axis_offset;
    double surface_axis_offset_m;
    int32_t enable_stripe_classification;
    int32_t has_refl;
};

struct ClusterDescriptorC {
    double cx;
    double cy;
    double cz;
    double var_x;
    double cov_xy;
    double var_y;
    double var_z;
    double geom_quality;
    double stripe_ratio;
    int32_t point_count;
};

}  // namespace fusion_core_cpp

FUSION_CPP_EXPORT int32_t cpp_extract_and_filter_ground(
    const uint8_t* raw_bytes,
    const fusion_core_cpp::LidarPipelineConfigC* cfg,
    double* out_x,
    double* out_y,
    double* out_z,
    double* out_refl,
    int32_t max_out,
    fusion_core_cpp::LidarExtractStatsC* out_stats
);

FUSION_CPP_EXPORT int32_t cpp_filter_ground_only(
    int32_t n_roi,
    const double* roi_x,
    const double* roi_y,
    const double* roi_z,
    const double* roi_refl,
    const fusion_core_cpp::LidarPipelineConfigC* cfg,
    double* out_x,
    double* out_y,
    double* out_z,
    double* out_refl,
    int32_t max_out
);

FUSION_CPP_EXPORT int32_t cpp_cluster_and_build(
    int32_t n_pts,
    const double* kx,
    const double* ky,
    const double* kz,
    const double* krefl,
    const fusion_core_cpp::ClusterBuildConfigC* cfg,
    fusion_core_cpp::ClusterDescriptorC* out_clusters,
    int32_t max_clusters
);
