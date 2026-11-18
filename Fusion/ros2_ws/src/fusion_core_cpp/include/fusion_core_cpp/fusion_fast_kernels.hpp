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

struct CameraIntrinsicsC {
    double fx;
    double fy;
    double cx;
    double cy;
    int32_t width;
    int32_t height;
    double k1;
    double k2;
    double p1;
    double p2;
    double k3;
};

struct ExtrinsicsC {
    double r[9];
    double t[3];
    double min_depth_m;
    double max_depth_m;
    double sample_radius_scale;
    double min_semantic_weight;
    double no_match_semantic_weight;
    int32_t soft_likelihood_enabled;
    double soft_likelihood_sigma_scale;
    double soft_likelihood_min_value;
    double soft_likelihood_outside_ratio;
    int32_t enable_frustum_hull_sampling;
    int32_t enable_triangular_cone_mask;
    double cone_mask_apex_width_ratio;
};

struct ClusterInputC {
    double cx;
    double cy;
    double cz;
    double cov_xx;
    double cov_yy;
    double cov_zz;
};

struct Detection2DC {
    double xmin;
    double ymin;
    double xmax;
    double ymax;
    double class_probs[4];
    double confidence;
    int32_t polygon_count;
    double polygon_u[16];
    double polygon_v[16];
    int32_t is_big_orange;
};

struct StereoDepthImageC {
    int32_t width;
    int32_t height;
    int32_t encoding;  // 0=none, 1=32FC1_meters, 2=16UC1_millimeters, 3=32FC1_disparity
    const uint8_t* data;
    int32_t row_step_bytes;
    double depth_scale;
    double baseline_m;
    double disparity_sigma_px;
    double min_depth_m;
    double max_depth_m;
};

struct FrustumRecoveryConfigC {
    double cone_height_m;
    double big_orange_height_m;
    double min_confidence;
    double min_bbox_height_px;
    double min_depth_m;
    double max_depth_m;
    double dedup_radius_m;
    double range_std_fraction;
    double bearing_std_fraction;
    int32_t enable_ground_plane_depth_fusion;
    double ground_z_base_m;
    double ground_slope_rad;
    int32_t enable_stereo_depth_fusion;
    int32_t enable_triangular_cone_mask;
    double cone_mask_apex_width_ratio;
};

struct RecoveredConeC {
    double x;
    double y;
    double z;
    double covariance[9];
    double class_probs[4];
    double confidence;
    double depth_m;
    int32_t detection_idx;
    int32_t used_stereo;
};

struct ClusterSemanticC {
    double class_probs[4];
    double semantic_weight;
    int32_t projected_samples;
    int32_t matched_samples;
    double matched_weight_sum;
    double best_confidence;
    int32_t residual_samples;
    double residual_sum_px;
    double residual_max_px;
    int32_t matched_detection_idx;
};

struct TrackerConfigC {
    double mahalanobis_gate;
    double semantic_min_overlap;
    double semantic_cost_weight;
    double process_noise_accel;
    double min_meas_var;
    int32_t confirm_hits;
    int32_t max_missed;
    double class_update_alpha;
    double max_predict_dt;
    int32_t motion_model_landmark;
    int32_t assignment_hungarian;
    int32_t semantic_dirichlet;
    double dirichlet_prior_strength;
    double dirichlet_evidence_scale;
    double dirichlet_max_concentration;
    double landmark_process_noise_pos;
    double odom_vx_std;
    double odom_vy_std;
    double odom_wz_std;
    double corridor_prior_weight;
    int32_t enable_global_map;
    double global_map_gate_radius_m;
    double global_map_min_lap_distance_m;
};

struct CorridorEstimateC {
    double xm;
    double y0;
    double slope;
    double curvature_kappa;
    double track_width_m;
    double confidence;
    int32_t n_waypoints;
    double left_x[64];
    double left_y[64];
    double right_x[64];
    double right_y[64];
    double wp_x[64];
    double wp_y[64];
    double wp_heading[64];
    double wp_kappa[64];
};

struct OnlineCalibPairC {
    double cx;
    double cy;
    double cz;
    double det_u;
    double det_v;
    double det_w;
    double det_h;
};

struct OnlineCalibResultC {
    int32_t matched_pairs;
    double initial_rmse_px;
    double final_rmse_px;
    double delta_roll_rad;
    double delta_pitch_rad;
    double delta_yaw_rad;
    double refined_r[9];
    double refined_t[3];
};

struct Observation2DC {
    double x;
    double y;
    double z;
    double cov_xx;
    double cov_xy;
    double cov_yy;
    double z_var;
    double class_probs[4];
    double quality;
};

struct EgoMotionC {
    int32_t has_motion;
    double vx;
    double vy;
    double wz;
    double dt;
    int32_t has_disp;
    double dx;
    double dy;
    double dyaw;
};

struct TrackEstimateC {
    int32_t track_id;
    double state[4];
    double covariance[16];
    double z;
    double z_var;
    double class_probs[4];
    int32_t hits;
    int32_t misses;
    double quality;
    int32_t status;
    double first_stamp;
    double last_stamp;
    double semantic_certainty;
};

struct TrackerStepStatsC {
    int32_t tracks_created;
    int32_t tracks_deleted;
    int32_t assoc_candidates;
    int32_t assoc_matches;
    int32_t assoc_rejects;
    int32_t n_lost_tracks;
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

FUSION_CPP_EXPORT int32_t cpp_estimate_batch_semantics(
    int32_t n_clusters,
    const fusion_core_cpp::ClusterInputC* clusters,
    int32_t n_dets,
    const fusion_core_cpp::Detection2DC* dets,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    const double* fallback_probs4,
    fusion_core_cpp::ClusterSemanticC* out_semantics
);

FUSION_CPP_EXPORT void* cpp_tracker_create(
    const fusion_core_cpp::TrackerConfigC* cfg
);

FUSION_CPP_EXPORT void cpp_tracker_destroy(void* handle);

FUSION_CPP_EXPORT void cpp_tracker_reset(void* handle);

FUSION_CPP_EXPORT int32_t cpp_tracker_step(
    void* handle,
    double stamp,
    const fusion_core_cpp::EgoMotionC* ego,
    int32_t n_obs,
    const fusion_core_cpp::Observation2DC* obs,
    fusion_core_cpp::TrackEstimateC* out_tracks,
    int32_t max_tracks,
    fusion_core_cpp::TrackEstimateC* out_lost_tracks,
    int32_t max_lost_tracks,
    fusion_core_cpp::TrackerStepStatsC* out_stats
);

FUSION_CPP_EXPORT int32_t cpp_tracker_get_tracks(
    const void* handle,
    fusion_core_cpp::TrackEstimateC* out_tracks,
    int32_t max_tracks
);

FUSION_CPP_EXPORT int32_t cpp_recover_unmatched_detections(
    int32_t n_dets,
    const fusion_core_cpp::Detection2DC* dets,
    const uint8_t* matched_det_flags,
    int32_t n_existing,
    const fusion_core_cpp::ClusterInputC* existing_positions,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    const fusion_core_cpp::FrustumRecoveryConfigC* cfg,
    const fusion_core_cpp::StereoDepthImageC* stereo_depth,
    fusion_core_cpp::RecoveredConeC* out_recovered,
    int32_t max_recovered
);

FUSION_CPP_EXPORT void* cpp_tracker_snapshot_create(const void* handle);

FUSION_CPP_EXPORT void cpp_tracker_snapshot_restore(void* handle, const void* snapshot);

FUSION_CPP_EXPORT void cpp_tracker_snapshot_destroy(void* snapshot);

FUSION_CPP_EXPORT double cpp_tracker_get_last_stamp(const void* handle);

FUSION_CPP_EXPORT int32_t cpp_tracker_get_corridor(
    const void* handle,
    double horizon_m,
    double step_m,
    const fusion_core_cpp::EgoMotionC* ego,
    fusion_core_cpp::CorridorEstimateC* out_corridor
);

FUSION_CPP_EXPORT int32_t cpp_online_calibrate_extrinsics(
    int32_t n_pairs,
    const fusion_core_cpp::OnlineCalibPairC* pairs,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    int32_t min_pairs,
    fusion_core_cpp::OnlineCalibResultC* out_res
);
