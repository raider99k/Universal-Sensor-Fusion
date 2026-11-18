#include "fusion_core_cpp/fusion_fast_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fusion_interfaces/msg/camera_cone_detections.hpp"
#include "fusion_interfaces/msg/cone_observation.hpp"
#include "fusion_interfaces/msg/cone_track.hpp"
#include "fusion_interfaces/msg/lidar_cone_cluster.hpp"
#include "fusion_interfaces/msg/lidar_cone_clusters.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace fusion_core_cpp {

namespace {

inline double stamp_to_sec(const builtin_interfaces::msg::Time& st) {
    return static_cast<double>(st.sec) + static_cast<double>(st.nanosec) * 1e-9;
}

inline void build_rpy_rotation(double roll, double pitch, double yaw, double r_out[9]) {
    const double cr = std::cos(roll), sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cy = std::cos(yaw), sy = std::sin(yaw);

    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r_out[0] = cy * cp;
    r_out[1] = cy * sp * sr - sy * cr;
    r_out[2] = cy * sp * cr + sy * sr;
    r_out[3] = sy * cp;
    r_out[4] = sy * sp * sr + cy * cr;
    r_out[5] = sy * sp * cr - cy * sr;
    r_out[6] = -sp;
    r_out[7] = cp * sr;
    r_out[8] = cp * cr;
}

}  // namespace

class OusterLidarComposableNode : public rclcpp::Node {
public:
    explicit OusterLidarComposableNode(const rclcpp::NodeOptions& options)
        : rclcpp::Node("ouster_lidar_cpp_adapter", options) {
        input_topic_ = this->declare_parameter<std::string>("input_topic", "/ouster/points");
        output_topic_ = this->declare_parameter<std::string>("output_topic", "/lidar/cone_clusters");
        motion_topic_ = this->declare_parameter<std::string>("motion_topic", "/ego/odometry");
        imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/imu/data");
        target_frame_ = this->declare_parameter<std::string>("target_frame", "base_link");

        pipe_cfg_.sensor_yaw_rad = this->declare_parameter("sensor_yaw_rad", 0.0);
        pipe_cfg_.sensor_offset_x_m = this->declare_parameter("sensor_offset_x_m", 0.0);
        pipe_cfg_.sensor_offset_y_m = this->declare_parameter("sensor_offset_y_m", 0.0);
        pipe_cfg_.sensor_offset_z_m = this->declare_parameter("sensor_offset_z_m", 0.0);
        pipe_cfg_.enable_imu_leveling = this->declare_parameter("enable_imu_leveling", true) ? 1 : 0;
        max_leveling_angle_rad_ = this->declare_parameter("max_leveling_angle_rad", 0.12);

        pipe_cfg_.roi_x_min = this->declare_parameter("roi_x_min", 0.0);
        pipe_cfg_.roi_x_max = this->declare_parameter("roi_x_max", 35.0);
        pipe_cfg_.roi_y_min = this->declare_parameter("roi_y_min", -12.0);
        pipe_cfg_.roi_y_max = this->declare_parameter("roi_y_max", 12.0);
        pipe_cfg_.roi_z_min = this->declare_parameter("roi_z_min", -0.60);
        pipe_cfg_.roi_z_max = this->declare_parameter("roi_z_max", 1.20);

        pipe_cfg_.enable_self_return_suppression =
            this->declare_parameter("enable_self_return_suppression", true) ? 1 : 0;
        pipe_cfg_.self_return_min_range_m = this->declare_parameter("self_return_min_range_m", 1.0);
        pipe_cfg_.self_return_radius_m = this->declare_parameter("self_return_radius_m", 1.7);
        pipe_cfg_.self_return_z_min_m = this->declare_parameter("self_return_z_min_m", -1.2);
        pipe_cfg_.self_return_z_max_m = this->declare_parameter("self_return_z_max_m", 1.0);

        pipe_cfg_.max_points_for_clustering = this->declare_parameter("max_points_for_clustering", 6000);
        max_clusters_ = this->declare_parameter("max_clusters", 128);
        enable_motion_deskew_ = this->declare_parameter("enable_motion_deskew", false);
        pipe_cfg_.deskew_scan_duration_s = this->declare_parameter("deskew_scan_duration_s", 0.10);
        pipe_cfg_.deskew_max_dt_s = this->declare_parameter("deskew_max_dt_s", 0.20);
        pipe_cfg_.deskew_spin_direction = this->declare_parameter("deskew_spin_direction", 1.0);

        pipe_cfg_.enable_adaptive_ground_model =
            this->declare_parameter("enable_adaptive_ground_model", true) ? 1 : 0;
        pipe_cfg_.ground_z_max = this->declare_parameter("ground_z_max", -0.08);
        pipe_cfg_.ground_zone_radial_step_m = this->declare_parameter("ground_zone_radial_step_m", 3.5);
        pipe_cfg_.ground_zone_azimuth_bins = this->declare_parameter("ground_zone_azimuth_bins", 12);
        pipe_cfg_.max_ground_slope_rad = this->declare_parameter("max_ground_slope_rad", 0.18);
        pipe_cfg_.ground_clearance_m = this->declare_parameter("ground_clearance_m", 0.05);
        pipe_cfg_.cone_max_radius_m = this->declare_parameter("cone_max_radius_m", 0.40);
        pipe_cfg_.enable_ground_point_recovery =
            this->declare_parameter("enable_ground_point_recovery", true) ? 1 : 0;
        pipe_cfg_.adaptive_range_ref_m = this->declare_parameter("adaptive_range_ref_m", 12.0);
        pipe_cfg_.cluster_tolerance_m = this->declare_parameter("cluster_tolerance_m", 0.40);

        cls_cfg_.cluster_tolerance_m = pipe_cfg_.cluster_tolerance_m;
        cls_cfg_.min_cluster_points = this->declare_parameter("min_cluster_points", 4);
        cls_cfg_.max_cluster_points = this->declare_parameter("max_cluster_points", 120);
        cls_cfg_.enable_adaptive_clustering =
            this->declare_parameter("enable_adaptive_clustering", true) ? 1 : 0;
        cls_cfg_.min_cluster_points_far = this->declare_parameter("min_cluster_points_far", 2);
        cls_cfg_.adaptive_range_ref_m = pipe_cfg_.adaptive_range_ref_m;
        cls_cfg_.cone_min_height_m = this->declare_parameter("cone_min_height_m", 0.05);
        cls_cfg_.cone_max_height_m = this->declare_parameter("cone_max_height_m", 0.80);
        cls_cfg_.cone_max_radius_m = pipe_cfg_.cone_max_radius_m;
        cls_cfg_.covariance_floor_xy = this->declare_parameter("covariance_floor_xy", 0.03);
        cls_cfg_.covariance_floor_z = this->declare_parameter("covariance_floor_z", 0.05);
        cls_cfg_.enable_surface_axis_offset =
            this->declare_parameter("enable_surface_axis_offset", true) ? 1 : 0;
        cls_cfg_.surface_axis_offset_m = this->declare_parameter("surface_axis_offset_m", 0.045);
        cls_cfg_.enable_stripe_classification =
            this->declare_parameter("enable_stripe_classification", true) ? 1 : 0;
        stripe_blue_min_ratio_ = this->declare_parameter("stripe_blue_min_ratio", 1.20);
        stripe_yellow_max_ratio_ = this->declare_parameter("stripe_yellow_max_ratio", 0.75);

        pub_clusters_ = this->create_publisher<fusion_interfaces::msg::LidarConeClusters>(
            output_topic_, rclcpp::QoS(10));

        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->on_cloud(msg); });

        sub_motion_ = this->create_subscription<nav_msgs::msg::Odometry>(
            motion_topic_, rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->on_motion(msg); });

        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { this->on_imu(msg); });
    }

private:
    void on_motion(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
        has_motion_ = true;
        motion_vx_ = msg->twist.twist.linear.x;
        motion_vy_ = msg->twist.twist.linear.y;
        motion_wz_ = msg->twist.twist.angular.z;
    }

    void on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
        const auto& q = msg->orientation;
        const double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        const double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        const double roll = std::atan2(sinr_cosp, cosr_cosp);
        const double sinp = std::max(-1.0, std::min(1.0, 2.0 * (q.w * q.y - q.z * q.x)));
        const double pitch = std::asin(sinp);
        if (std::abs(roll) <= max_leveling_angle_rad_ && std::abs(pitch) <= max_leveling_angle_rad_) {
            sin_roll_ = std::sin(roll);
            sin_pitch_ = std::sin(pitch);
        }
    }

    void on_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        pipe_cfg_.n_total = static_cast<int32_t>(msg->width * msg->height);
        pipe_cfg_.height = static_cast<int32_t>(msg->height);
        pipe_cfg_.width = static_cast<int32_t>(msg->width);
        pipe_cfg_.point_step = static_cast<int32_t>(msg->point_step);
        pipe_cfg_.off_x = 0;
        pipe_cfg_.off_y = 4;
        pipe_cfg_.off_z = 8;
        pipe_cfg_.off_t = -1;
        pipe_cfg_.off_stripe = -1;
        pipe_cfg_.has_motion = (enable_motion_deskew_ && has_motion_) ? 1 : 0;
        pipe_cfg_.vx = motion_vx_;
        pipe_cfg_.vy = motion_vy_;
        pipe_cfg_.wz = motion_wz_;
        pipe_cfg_.sin_pitch = sin_pitch_;
        pipe_cfg_.sin_roll = sin_roll_;

        for (const auto& f : msg->fields) {
            if (f.name == "x") pipe_cfg_.off_x = static_cast<int32_t>(f.offset);
            else if (f.name == "y") pipe_cfg_.off_y = static_cast<int32_t>(f.offset);
            else if (f.name == "z") pipe_cfg_.off_z = static_cast<int32_t>(f.offset);
            else if (f.name == "t" || f.name == "time") {
                pipe_cfg_.off_t = static_cast<int32_t>(f.offset);
                pipe_cfg_.t_datatype = static_cast<int32_t>(f.datatype);
            } else if (f.name == "reflectivity" || f.name == "intensity") {
                pipe_cfg_.off_stripe = static_cast<int32_t>(f.offset);
                pipe_cfg_.stripe_datatype = static_cast<int32_t>(f.datatype);
            }
        }

        const int32_t max_pts = std::max(512, pipe_cfg_.max_points_for_clustering);
        buf_x_.resize(static_cast<size_t>(max_pts));
        buf_y_.resize(static_cast<size_t>(max_pts));
        buf_z_.resize(static_cast<size_t>(max_pts));
        buf_refl_.resize(static_cast<size_t>(max_pts));
        LidarExtractStatsC stats{};
        const int32_t n_kept = cpp_extract_and_filter_ground(
            msg->data.data(), &pipe_cfg_,
            buf_x_.data(), buf_y_.data(), buf_z_.data(), buf_refl_.data(),
            max_pts, &stats);

        cls_cfg_.has_refl = (pipe_cfg_.off_stripe >= 0) ? 1 : 0;
        const int32_t max_c = std::max(1, max_clusters_);
        std::vector<ClusterDescriptorC> desc(static_cast<size_t>(max_c));
        const int32_t n_cls = cpp_cluster_and_build(
            n_kept, buf_x_.data(), buf_y_.data(), buf_z_.data(),
            cls_cfg_.has_refl ? buf_refl_.data() : nullptr,
            &cls_cfg_, desc.data(), max_c);

        auto out_msg = std::make_unique<fusion_interfaces::msg::LidarConeClusters>();
        out_msg->header = msg->header;
        if (!target_frame_.empty()) {
            out_msg->header.frame_id = target_frame_;
        }
        out_msg->clusters.reserve(static_cast<size_t>(n_cls));
        for (int32_t i = 0; i < n_cls; ++i) {
            fusion_interfaces::msg::LidarConeCluster c;
            c.centroid.x = desc[static_cast<size_t>(i)].cx;
            c.centroid.y = desc[static_cast<size_t>(i)].cy;
            c.centroid.z = desc[static_cast<size_t>(i)].cz;
            c.covariance[0] = desc[static_cast<size_t>(i)].var_x;
            c.covariance[1] = desc[static_cast<size_t>(i)].cov_xy;
            c.covariance[3] = desc[static_cast<size_t>(i)].cov_xy;
            c.covariance[4] = desc[static_cast<size_t>(i)].var_y;
            c.covariance[8] = desc[static_cast<size_t>(i)].var_z;
            c.geometry_quality = static_cast<float>(desc[static_cast<size_t>(i)].geom_quality);
            c.point_count = static_cast<uint32_t>(desc[static_cast<size_t>(i)].point_count);
            if (cls_cfg_.enable_stripe_classification != 0 && cls_cfg_.has_refl != 0) {
                const double ratio = desc[static_cast<size_t>(i)].stripe_ratio;
                if (ratio >= stripe_blue_min_ratio_) {
                    c.stripe_class_probs = {0.72f, 0.10f, 0.10f, 0.08f};
                } else if (ratio <= stripe_yellow_max_ratio_) {
                    c.stripe_class_probs = {0.10f, 0.72f, 0.10f, 0.08f};
                } else {
                    c.stripe_class_probs = {0.25f, 0.25f, 0.25f, 0.25f};
                }
            } else {
                c.stripe_class_probs = {0.25f, 0.25f, 0.25f, 0.25f};
            }
            out_msg->clusters.push_back(c);
        }
        pub_clusters_->publish(std::move(out_msg));
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string motion_topic_;
    std::string imu_topic_;
    std::string target_frame_;
    double max_leveling_angle_rad_{0.12};
    double stripe_blue_min_ratio_{1.20};
    double stripe_yellow_max_ratio_{0.75};
    int32_t max_clusters_{128};
    bool enable_motion_deskew_{false};
    bool has_motion_{false};
    double motion_vx_{0.0};
    double motion_vy_{0.0};
    double motion_wz_{0.0};
    double sin_pitch_{0.0};
    double sin_roll_{0.0};
    LidarPipelineConfigC pipe_cfg_{};
    ClusterBuildConfigC cls_cfg_{};
    std::vector<double> buf_x_;
    std::vector<double> buf_y_;
    std::vector<double> buf_z_;
    std::vector<double> buf_refl_;
    rclcpp::Publisher<fusion_interfaces::msg::LidarConeClusters>::SharedPtr pub_clusters_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_motion_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
};

class FusionCoreComposableNode : public rclcpp::Node {
public:
    explicit FusionCoreComposableNode(const rclcpp::NodeOptions& options)
        : rclcpp::Node("fusion_core_cpp_node", options) {
        camera_topic_ = this->declare_parameter<std::string>("camera_topic", "/camera/cone_detections");
        lidar_topic_ = this->declare_parameter<std::string>("lidar_topic", "/lidar/cone_clusters");
        motion_topic_ = this->declare_parameter<std::string>("motion_topic", "/ego/odometry");
        obs_out_topic_ = this->declare_parameter<std::string>("obs_out_topic", "/perception/cone_observations");
        track_out_topic_ = this->declare_parameter<std::string>("track_out_topic", "/perception/cone_tracks");
        diag_topic_ = this->declare_parameter<std::string>("diag_topic", "/perception/diagnostics");
        target_frame_ = this->declare_parameter<std::string>("target_frame", "base_link");

        max_camera_lidar_delta_s_ = this->declare_parameter("max_camera_lidar_delta_ms", 120.0) * 1e-3;
        diag_rate_hz_ = this->declare_parameter("diag_rate_hz", 2.0);

        intr_.fx = this->declare_parameter("camera_fx", 700.0);
        intr_.fy = this->declare_parameter("camera_fy", 700.0);
        intr_.cx = this->declare_parameter("camera_cx", 640.0);
        intr_.cy = this->declare_parameter("camera_cy", 360.0);
        intr_.width = this->declare_parameter("camera_width", 1280);
        intr_.height = this->declare_parameter("camera_height", 720);
        intr_.k1 = this->declare_parameter("camera_k1", 0.0);
        intr_.k2 = this->declare_parameter("camera_k2", 0.0);
        intr_.p1 = this->declare_parameter("camera_p1", 0.0);
        intr_.p2 = this->declare_parameter("camera_p2", 0.0);
        intr_.k3 = this->declare_parameter("camera_k3", 0.0);

        const double tx = this->declare_parameter("lidar_to_camera_tx_m", 0.0);
        const double ty = this->declare_parameter("lidar_to_camera_ty_m", 0.0);
        const double tz = this->declare_parameter("lidar_to_camera_tz_m", 0.0);
        const double roll = this->declare_parameter("lidar_to_camera_roll_rad", -3.141592653589793);
        const double pitch = this->declare_parameter("lidar_to_camera_pitch_rad", -1.5707963267948966);
        const double yaw = this->declare_parameter("lidar_to_camera_yaw_rad", -1.5707963267948966);
        build_rpy_rotation(roll, pitch, yaw, ext_.r);
        ext_.t[0] = tx;
        ext_.t[1] = ty;
        ext_.t[2] = tz;
        ext_.min_depth_m = this->declare_parameter("projection_min_depth_m", 0.1);
        ext_.max_depth_m = this->declare_parameter("projection_max_depth_m", 80.0);
        ext_.sample_radius_scale = this->declare_parameter("projection_sample_radius_scale", 1.5);
        ext_.min_semantic_weight = this->declare_parameter("projection_min_semantic_weight", 0.35);
        ext_.no_match_semantic_weight = this->declare_parameter("projection_no_match_semantic_weight", 0.60);
        ext_.soft_likelihood_enabled =
            this->declare_parameter("projection_soft_likelihood_enabled", true) ? 1 : 0;
        ext_.soft_likelihood_sigma_scale = this->declare_parameter("projection_soft_likelihood_sigma_scale", 0.35);
        ext_.soft_likelihood_min_value = this->declare_parameter("projection_soft_likelihood_min_value", 0.02);
        ext_.soft_likelihood_outside_ratio = this->declare_parameter("projection_soft_likelihood_outside_ratio", 1.6);
        ext_.enable_frustum_hull_sampling =
            this->declare_parameter("enable_frustum_hull_sampling", false) ? 1 : 0;

        trk_cfg_.mahalanobis_gate = this->declare_parameter("track_mahalanobis_gate", 9.21);
        trk_cfg_.semantic_min_overlap = this->declare_parameter("track_semantic_min_overlap", 0.15);
        trk_cfg_.semantic_cost_weight = this->declare_parameter("track_semantic_cost_weight", 2.0);
        trk_cfg_.process_noise_accel = this->declare_parameter("track_process_noise_accel", 2.0);
        trk_cfg_.min_meas_var = this->declare_parameter("track_min_meas_var", 0.05);
        trk_cfg_.confirm_hits = this->declare_parameter("track_confirm_hits", 3);
        trk_cfg_.max_missed = this->declare_parameter("track_max_missed", 5);
        trk_cfg_.class_update_alpha = this->declare_parameter("track_class_update_alpha", 0.35);
        trk_cfg_.max_predict_dt = this->declare_parameter("track_max_predict_dt_s", 0.25);
        trk_cfg_.motion_model_landmark =
            (this->declare_parameter<std::string>("track_motion_model", "landmark") == "landmark") ? 1 : 0;
        trk_cfg_.assignment_hungarian =
            (this->declare_parameter<std::string>("track_assignment_algorithm", "hungarian") == "hungarian") ? 1 : 0;
        trk_cfg_.semantic_dirichlet =
            (this->declare_parameter<std::string>("track_semantic_model", "dirichlet") == "dirichlet") ? 1 : 0;
        trk_cfg_.dirichlet_prior_strength = this->declare_parameter("track_dirichlet_prior_strength", 1.0);
        trk_cfg_.dirichlet_evidence_scale = this->declare_parameter("track_dirichlet_evidence_scale", 4.0);
        trk_cfg_.dirichlet_max_concentration = this->declare_parameter("track_dirichlet_max_concentration", 40.0);
        trk_cfg_.landmark_process_noise_pos = this->declare_parameter("track_landmark_process_noise_pos", 0.03);
        trk_cfg_.odom_vx_std = this->declare_parameter("track_odom_vx_std", 0.15);
        trk_cfg_.odom_vy_std = this->declare_parameter("track_odom_vy_std", 0.15);
        trk_cfg_.odom_wz_std = this->declare_parameter("track_odom_wz_std", 0.05);

        tracker_handle_ = cpp_tracker_create(&trk_cfg_);

        pub_obs_ = this->create_publisher<fusion_interfaces::msg::ConeObservation>(
            obs_out_topic_, rclcpp::QoS(20));
        pub_tracks_ = this->create_publisher<fusion_interfaces::msg::ConeTrack>(
            track_out_topic_, rclcpp::QoS(20));
        pub_diag_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            diag_topic_, rclcpp::QoS(10));

        sub_cam_ = this->create_subscription<fusion_interfaces::msg::CameraConeDetections>(
            camera_topic_, rclcpp::SensorDataQoS(),
            [this](fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr msg) {
                camera_buffer_.push_back(msg);
                while (camera_buffer_.size() > 16) camera_buffer_.pop_front();
            });
        sub_motion_ = this->create_subscription<nav_msgs::msg::Odometry>(
            motion_topic_, rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                latest_motion_ = msg;
            });
        sub_lidar_ = this->create_subscription<fusion_interfaces::msg::LidarConeClusters>(
            lidar_topic_, rclcpp::SensorDataQoS(),
            [this](fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr msg) {
                this->on_lidar(msg);
            });

        const double diag_period_s = 1.0 / std::max(0.5, diag_rate_hz_);
        diag_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(diag_period_s)),
            [this]() { this->publish_diagnostics(); });
    }

    ~FusionCoreComposableNode() override {
        if (tracker_handle_) {
            cpp_tracker_destroy(tracker_handle_);
            tracker_handle_ = nullptr;
        }
    }

private:
    void on_lidar(const fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr& lidar_msg) {
        const double lidar_stamp = stamp_to_sec(lidar_msg->header.stamp);
        fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr cam_msg;
        double best_dt = 1e9;
        for (const auto& candidate : camera_buffer_) {
            const double dt = std::abs(stamp_to_sec(candidate->header.stamp) - lidar_stamp);
            if (dt < best_dt) {
                best_dt = dt;
                cam_msg = candidate;
            }
        }
        if (cam_msg && best_dt > max_camera_lidar_delta_s_) {
            cam_msg.reset();
        }

        const int32_t n_cls = static_cast<int32_t>(lidar_msg->clusters.size());
        std::vector<ClusterInputC> cls_in(static_cast<size_t>(n_cls));
        for (int32_t i = 0; i < n_cls; ++i) {
            const auto& c = lidar_msg->clusters[static_cast<size_t>(i)];
            cls_in[static_cast<size_t>(i)] = {
                c.centroid.x, c.centroid.y, c.centroid.z,
                c.covariance[0], c.covariance[4], c.covariance[8]
            };
        }

        std::vector<Detection2DC> dets_in;
        if (cam_msg) {
            dets_in.reserve(cam_msg->detections.size());
            for (const auto& d : cam_msg->detections) {
                dets_in.push_back({
                    d.xmin, d.ymin, d.xmax, d.ymax,
                    {d.class_probs[0], d.class_probs[1], d.class_probs[2], d.class_probs[3]},
                    d.confidence
                });
            }
        }

        const std::array<double, 4> fallback{0.25, 0.25, 0.25, 0.25};
        std::vector<ClusterSemanticC> sem_out(static_cast<size_t>(n_cls));
        if (n_cls > 0) {
            cpp_estimate_batch_semantics(
                n_cls, cls_in.data(),
                static_cast<int32_t>(dets_in.size()),
                dets_in.empty() ? nullptr : dets_in.data(),
                &intr_, &ext_, fallback.data(), sem_out.data());
        }

        const std::string out_frame = target_frame_.empty() ? lidar_msg->header.frame_id : target_frame_;
        std::vector<Observation2DC> obs_c(static_cast<size_t>(n_cls));
        for (int32_t i = 0; i < n_cls; ++i) {
            const auto& c = lidar_msg->clusters[static_cast<size_t>(i)];
            const auto& s = sem_out[static_cast<size_t>(i)];
            const double geom_q = (c.geometry_quality > 0.0f) ? static_cast<double>(c.geometry_quality) : 1.0;
            const double q = std::max(0.0, std::min(1.0, geom_q * s.semantic_weight));

            fusion_interfaces::msg::ConeObservation obs_msg;
            obs_msg.header.stamp = lidar_msg->header.stamp;
            obs_msg.header.frame_id = out_frame;
            obs_msg.position = c.centroid;
            obs_msg.covariance = c.covariance;
            obs_msg.class_probs = {
                static_cast<float>(s.class_probs[0]),
                static_cast<float>(s.class_probs[1]),
                static_cast<float>(s.class_probs[2]),
                static_cast<float>(s.class_probs[3])
            };
            obs_msg.quality_score = static_cast<float>(q);
            obs_msg.sensor_health = 0;
            obs_msg.source = "cpp_live";
            pub_obs_->publish(obs_msg);

            obs_c[static_cast<size_t>(i)] = {
                c.centroid.x, c.centroid.y, c.centroid.z,
                c.covariance[0], c.covariance[1], c.covariance[4], c.covariance[8],
                {s.class_probs[0], s.class_probs[1], s.class_probs[2], s.class_probs[3]},
                q
            };
        }

        EgoMotionC ego{};
        if (latest_motion_) {
            ego.has_motion = 1;
            ego.vx = latest_motion_->twist.twist.linear.x;
            ego.vy = latest_motion_->twist.twist.linear.y;
            ego.wz = latest_motion_->twist.twist.angular.z;
            ego.dt = (last_stamp_ >= 0.0) ? std::max(0.0, lidar_stamp - last_stamp_) : 0.0;
        }
        last_stamp_ = lidar_stamp;

        std::array<TrackEstimateC, 256> active_tracks{};
        std::array<TrackEstimateC, 64> lost_tracks{};
        TrackerStepStatsC stats{};
        const int32_t n_tracks = cpp_tracker_step(
            tracker_handle_, lidar_stamp, &ego,
            n_cls, obs_c.empty() ? nullptr : obs_c.data(),
            active_tracks.data(), 256, lost_tracks.data(), 64, &stats);

        auto publish_track = [&](const TrackEstimateC& tr, uint8_t force_status) {
            fusion_interfaces::msg::ConeTrack msg;
            msg.header.stamp = lidar_msg->header.stamp;
            msg.header.frame_id = out_frame;
            msg.track_id = static_cast<uint32_t>(tr.track_id);
            msg.position.x = tr.state[0];
            msg.position.y = tr.state[1];
            msg.position.z = tr.z;
            msg.velocity.x = tr.state[2];
            msg.velocity.y = tr.state[3];
            msg.velocity.z = 0.0;
            msg.covariance[0] = tr.covariance[0];
            msg.covariance[1] = tr.covariance[1];
            msg.covariance[3] = tr.covariance[4];
            msg.covariance[4] = tr.covariance[5];
            msg.covariance[8] = tr.z_var;
            msg.class_probs = {
                static_cast<float>(tr.class_probs[0]),
                static_cast<float>(tr.class_probs[1]),
                static_cast<float>(tr.class_probs[2]),
                static_cast<float>(tr.class_probs[3])
            };
            msg.track_age = static_cast<float>(std::max(0.0, tr.last_stamp - tr.first_stamp));
            msg.track_status = force_status;
            msg.quality_score = static_cast<float>(tr.quality);
            pub_tracks_->publish(msg);
        };

        const int32_t n_lost = std::min<int32_t>(stats.n_lost_tracks, 64);
        for (int32_t i = 0; i < n_lost; ++i) {
            publish_track(lost_tracks[static_cast<size_t>(i)], fusion_interfaces::msg::ConeTrack::TRACK_LOST);
        }
        for (int32_t i = 0; i < n_tracks; ++i) {
            const auto& tr = active_tracks[static_cast<size_t>(i)];
            publish_track(tr, static_cast<uint8_t>(tr.status));
        }

        last_obs_count_ = n_cls;
        last_track_count_ = n_tracks;
        ++cycles_completed_;
    }

    void publish_diagnostics() {
        diagnostic_msgs::msg::DiagnosticArray msg;
        msg.header.stamp = this->now();
        diagnostic_msgs::msg::DiagnosticStatus st;
        st.name = "fusion_core_cpp/runtime";
        st.hardware_id = "fusion_core_cpp";
        st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        st.message = "ok";

        auto add_kv = [&](const std::string& k, const std::string& v) {
            diagnostic_msgs::msg::KeyValue kv;
            kv.key = k;
            kv.value = v;
            st.values.push_back(kv);
        };
        add_kv("cycles_completed", std::to_string(cycles_completed_));
        add_kv("last_observation_count", std::to_string(last_obs_count_));
        add_kv("tracks_active", std::to_string(last_track_count_));
        msg.status.push_back(st);
        pub_diag_->publish(msg);
    }

    std::string camera_topic_;
    std::string lidar_topic_;
    std::string motion_topic_;
    std::string obs_out_topic_;
    std::string track_out_topic_;
    std::string diag_topic_;
    std::string target_frame_;
    double max_camera_lidar_delta_s_{0.12};
    double diag_rate_hz_{2.0};
    double last_stamp_{-1.0};
    int32_t last_obs_count_{0};
    int32_t last_track_count_{0};
    int64_t cycles_completed_{0};

    CameraIntrinsicsC intr_{};
    ExtrinsicsC ext_{};
    TrackerConfigC trk_cfg_{};
    void* tracker_handle_{nullptr};

    std::deque<fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr> camera_buffer_;
    nav_msgs::msg::Odometry::ConstSharedPtr latest_motion_;

    rclcpp::Publisher<fusion_interfaces::msg::ConeObservation>::SharedPtr pub_obs_;
    rclcpp::Publisher<fusion_interfaces::msg::ConeTrack>::SharedPtr pub_tracks_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_;
    rclcpp::Subscription<fusion_interfaces::msg::CameraConeDetections>::SharedPtr sub_cam_;
    rclcpp::Subscription<fusion_interfaces::msg::LidarConeClusters>::SharedPtr sub_lidar_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_motion_;
    rclcpp::TimerBase::SharedPtr diag_timer_;
};

}  // namespace fusion_core_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(fusion_core_cpp::OusterLidarComposableNode)
RCLCPP_COMPONENTS_REGISTER_NODE(fusion_core_cpp::FusionCoreComposableNode)
