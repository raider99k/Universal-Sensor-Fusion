#include "fusion_core_cpp/fusion_fast_kernels.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fusion_interfaces/msg/camera_cone_detections.hpp"
#include "fusion_interfaces/msg/cone_observation.hpp"
#include "fusion_interfaces/msg/cone_track.hpp"
#include "fusion_interfaces/msg/lidar_cone_cluster.hpp"
#include "fusion_interfaces/msg/lidar_cone_clusters.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/string.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#ifdef FUSION_HAS_TF2_ROS
#if __has_include("tf2/exceptions.hpp")
#include "tf2/exceptions.hpp"
#else
#include "tf2/exceptions.h"
#endif
#if __has_include("tf2_ros/buffer.hpp")
#include "tf2_ros/buffer.hpp"
#else
#include "tf2_ros/buffer.h"
#endif
#if __has_include("tf2_ros/transform_listener.hpp")
#include "tf2_ros/transform_listener.hpp"
#else
#include "tf2_ros/transform_listener.h"
#endif
#endif

namespace fusion_core_cpp {

namespace {

constexpr int32_t kSensorHealthOk = 0;
constexpr int32_t kSensorHealthDegraded = 1;
constexpr int32_t kSensorHealthCritical = 2;

inline double stamp_to_sec(const builtin_interfaces::msg::Time& st) {
    return static_cast<double>(st.sec) + static_cast<double>(st.nanosec) * 1e-9;
}

inline double clamp_val(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
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

inline void quat_to_rotation_matrix(double x, double y, double z, double w, double r_out[9]) {
    const double n = x * x + y * y + z * z + w * w;
    if (n <= 1e-12) {
        r_out[0] = 1.0; r_out[1] = 0.0; r_out[2] = 0.0;
        r_out[3] = 0.0; r_out[4] = 1.0; r_out[5] = 0.0;
        r_out[6] = 0.0; r_out[7] = 0.0; r_out[8] = 1.0;
        return;
    }
    const double s = 2.0 / n;
    const double xx = x * x * s, yy = y * y * s, zz = z * z * s;
    const double xy = x * y * s, xz = x * z * s, yz = y * z * s;
    const double wx = w * x * s, wy = w * y * s, wz = w * z * s;
    r_out[0] = 1.0 - (yy + zz); r_out[1] = xy - wz;         r_out[2] = xz + wy;
    r_out[3] = xy + wz;         r_out[4] = 1.0 - (xx + zz); r_out[5] = yz - wx;
    r_out[6] = xz - wy;         r_out[7] = yz + wx;         r_out[8] = 1.0 - (xx + yy);
}

inline void mat_mul_3x3(const double a[9], const double b[9], double out[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[r * 3 + c] = a[r * 3 + 0] * b[0 + c] +
                             a[r * 3 + 1] * b[3 + c] +
                             a[r * 3 + 2] * b[6 + c];
        }
    }
}

inline void normalize_probs4_f(const std::array<float, 4>& in, std::array<double, 4>& out) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        out[i] = (std::isfinite(in[i]) && in[i] > 0.0f) ? static_cast<double>(in[i]) : 0.0;
        s += out[i];
    }
    if (s <= 1e-9) {
        out = {0.25, 0.25, 0.25, 0.25};
        return;
    }
    const double inv = 1.0 / s;
    for (int i = 0; i < 4; ++i) out[i] *= inv;
}

// Lightweight flat JSON object-list / object-dict parser for camera_profiles_json and lidar_profiles_json.
struct JsonProfileEntry {
    std::string key;
    std::map<std::string, std::string> strings;
    std::map<std::string, double> numbers;
    std::map<std::string, bool> bools;
};

inline void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
}

inline std::string parse_json_str(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') return "";
    ++pos;
    std::string out;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
        }
        out.push_back(s[pos++]);
    }
    if (pos < s.size() && s[pos] == '"') ++pos;
    return out;
}

inline void parse_json_obj(const std::string& s, size_t& pos, JsonProfileEntry& entry) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') return;
    ++pos;
    while (pos < s.size()) {
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] == '}') {
            if (pos < s.size()) ++pos;
            return;
        }
        if (s[pos] == ',') {
            ++pos;
            continue;
        }
        std::string k = parse_json_str(s, pos);
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ':') ++pos;
        skip_ws(s, pos);
        if (pos >= s.size()) break;
        if (s[pos] == '"') {
            entry.strings[k] = parse_json_str(s, pos);
        } else if (s.compare(pos, 4, "true") == 0) {
            entry.bools[k] = true;
            pos += 4;
        } else if (s.compare(pos, 5, "false") == 0) {
            entry.bools[k] = false;
            pos += 5;
        } else if (s.compare(pos, 4, "null") == 0) {
            pos += 4;
        } else {
            size_t start = pos;
            while (pos < s.size() && s[pos] != ',' && s[pos] != '}' &&
                   !std::isspace(static_cast<unsigned char>(s[pos]))) {
                ++pos;
            }
            try {
                entry.numbers[k] = std::stod(s.substr(start, pos - start));
            } catch (...) {
            }
        }
    }
}

inline std::vector<JsonProfileEntry> parse_profile_json(const std::string& raw) {
    std::vector<JsonProfileEntry> out;
    size_t pos = 0;
    skip_ws(raw, pos);
    if (pos >= raw.size()) return out;
    if (raw[pos] == '[') {
        ++pos;
        while (pos < raw.size()) {
            skip_ws(raw, pos);
            if (pos >= raw.size() || raw[pos] == ']') break;
            if (raw[pos] == ',') {
                ++pos;
                continue;
            }
            if (raw[pos] == '{') {
                JsonProfileEntry e;
                parse_json_obj(raw, pos, e);
                out.push_back(std::move(e));
            } else {
                break;
            }
        }
    } else if (raw[pos] == '{') {
        ++pos;
        while (pos < raw.size()) {
            skip_ws(raw, pos);
            if (pos >= raw.size() || raw[pos] == '}') break;
            if (raw[pos] == ',') {
                ++pos;
                continue;
            }
            std::string key = parse_json_str(raw, pos);
            skip_ws(raw, pos);
            if (pos < raw.size() && raw[pos] == ':') ++pos;
            skip_ws(raw, pos);
            if (pos < raw.size() && raw[pos] == '{') {
                JsonProfileEntry e;
                e.key = key;
                parse_json_obj(raw, pos, e);
                out.push_back(std::move(e));
            } else {
                break;
            }
        }
    }
    return out;
}

inline int32_t parse_distortion_model_c(const std::string& raw) {
    std::string lower;
    lower.reserve(raw.size());
    for (char c : raw) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "equidistant" || lower == "kannala_brandt" || lower == "fisheye") {
        return 1;
    }
    return 0;
}

inline std::string fmt_double(double v, int prec = 3) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return std::string(buf);
}

inline void compute_cov_ellipse_2sigma(double cxx, double cxy, double cyy,
                                       double& maj_d, double& min_d, double& yaw_rad) {
    const double trace = cxx + cyy;
    const double diff = cxx - cyy;
    const double disc = std::sqrt(std::max(0.0, diff * diff + 4.0 * cxy * cxy));
    const double eig1 = std::max(1e-6, 0.5 * (trace + disc));
    const double eig2 = std::max(1e-6, 0.5 * (trace - disc));
    yaw_rad = 0.5 * std::atan2(2.0 * cxy, diff);
    maj_d = 4.0 * std::sqrt(eig1);
    min_d = 4.0 * std::sqrt(eig2);
}

}  // namespace

class OusterLidarComposableNode : public rclcpp::Node {
public:
    explicit OusterLidarComposableNode(const rclcpp::NodeOptions& options)
        : rclcpp::Node("ouster_lidar_cpp_adapter", options) {
        input_topic_ = this->declare_parameter<std::string>("input_topic", "/ouster/points");
        output_topic_ = this->declare_parameter<std::string>("output_topic", "/lidar/cone_clusters");
        motion_topic_ = this->declare_parameter<std::string>("motion_topic", "/ego/odometry");
        motion_msg_type_ = this->declare_parameter<std::string>("motion_msg_type", "odometry");
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
        cls_cfg_.enable_cluster_splitting =
            this->declare_parameter("enable_cluster_splitting", true) ? 1 : 0;
        cls_cfg_.split_max_radius_m = this->declare_parameter("split_max_radius_m", 0.95);
        cls_cfg_.split_min_separation_m = this->declare_parameter("split_min_separation_m", 0.28);
        stripe_blue_min_ratio_ = this->declare_parameter("stripe_blue_min_ratio", 1.20);
        stripe_yellow_max_ratio_ = this->declare_parameter("stripe_yellow_max_ratio", 0.75);

        pub_clusters_ = this->create_publisher<fusion_interfaces::msg::LidarConeClusters>(
            output_topic_, rclcpp::QoS(10));

        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->on_cloud(msg); });

        if (motion_msg_type_ == "twist_stamped") {
            sub_motion_twist_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
                motion_topic_, rclcpp::SensorDataQoS(),
                [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
                    has_motion_ = true;
                    motion_vx_ = msg->twist.linear.x;
                    motion_vy_ = msg->twist.linear.y;
                    motion_wz_ = msg->twist.angular.z;
                });
        } else if (motion_msg_type_ == "twist_with_covariance_stamped") {
            sub_motion_twist_cov_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
                motion_topic_, rclcpp::SensorDataQoS(),
                [this](geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr msg) {
                    has_motion_ = true;
                    motion_vx_ = msg->twist.twist.linear.x;
                    motion_vy_ = msg->twist.twist.linear.y;
                    motion_wz_ = msg->twist.twist.angular.z;
                });
        } else {
            sub_motion_ = this->create_subscription<nav_msgs::msg::Odometry>(
                motion_topic_, rclcpp::SensorDataQoS(),
                [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->on_motion(msg); });
        }

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
    std::string motion_msg_type_{"odometry"};
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
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_motion_twist_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_motion_twist_cov_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
};

class FusionCoreComposableNode : public rclcpp::Node {
public:
    explicit FusionCoreComposableNode(const rclcpp::NodeOptions& options)
        : rclcpp::Node("fusion_core_cpp_node", options) {
        declare_and_load_parameters();
        build_registries();

        tracker_handle_ = cpp_tracker_create(&trk_cfg_);
        history_base_snapshot_ = cpp_tracker_snapshot_create(tracker_handle_);

        rclcpp::QoS sensor_qos(static_cast<size_t>(std::max(1, sensor_qos_depth_)));
        sensor_qos.best_effort().keep_last(static_cast<size_t>(std::max(1, sensor_qos_depth_))).durability_volatile();

        rclcpp::QoS output_qos(static_cast<size_t>(std::max(1, output_qos_depth_)));
        output_qos.reliable().keep_last(static_cast<size_t>(std::max(1, output_qos_depth_))).durability_volatile();

        rclcpp::QoS diag_qos(static_cast<size_t>(std::max(1, diag_qos_depth_)));
        diag_qos.reliable().keep_last(static_cast<size_t>(std::max(1, diag_qos_depth_))).durability_volatile();

        pub_obs_ = this->create_publisher<fusion_interfaces::msg::ConeObservation>(obs_out_topic_, output_qos);
        pub_tracks_ = this->create_publisher<fusion_interfaces::msg::ConeTrack>(track_out_topic_, output_qos);
        pub_diag_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(diag_topic_, diag_qos);
        pub_corridor_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            corridor_markers_topic_.empty() ? "/perception/corridor_markers" : corridor_markers_topic_, output_qos);
        pub_corridor_path_ = this->create_publisher<nav_msgs::msg::Path>(
            corridor_path_topic_.empty() ? "/perception/corridor_path" : corridor_path_topic_, output_qos);
        pub_debug_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            debug_markers_topic_.empty() ? "/perception/debug_markers" : debug_markers_topic_, output_qos);
        pub_debug_reprojection_ = this->create_publisher<std_msgs::msg::String>(
            debug_reprojection_topic_.empty() ? "/perception/debug_reprojection" : debug_reprojection_topic_, output_qos);

        for (size_t idx = 0; idx < camera_profiles_.size(); ++idx) {
            const auto& cp = camera_profiles_[idx];
            sub_cams_.push_back(
                this->create_subscription<fusion_interfaces::msg::CameraConeDetections>(
                    cp.topic, sensor_qos,
                    [this, idx](fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr msg) {
                        this->on_camera(msg, idx);
                    }));
            if (enable_camera_info_intrinsics_ && !cp.camera_info_topic.empty()) {
                sub_cam_infos_.push_back(
                    this->create_subscription<sensor_msgs::msg::CameraInfo>(
                        cp.camera_info_topic, sensor_qos,
                        [this, idx](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
                            this->on_camera_info(msg, idx);
                        }));
            }
            if (enable_stereo_depth_fusion_ && !cp.depth_topic.empty()) {
                sub_depths_.push_back(
                    this->create_subscription<sensor_msgs::msg::Image>(
                        cp.depth_topic, sensor_qos,
                        [this, idx](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                            this->on_depth(msg, idx);
                        }));
            }
        }

        sub_lidar_ = this->create_subscription<fusion_interfaces::msg::LidarConeClusters>(
            lidar_topic_, sensor_qos,
            [this](fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr msg) {
                this->on_lidar(msg);
            });

        for (size_t idx = 1; idx < lidar_profiles_.size(); ++idx) {
            const auto& lp = lidar_profiles_[idx];
            sub_extra_lidars_.push_back(
                this->create_subscription<fusion_interfaces::msg::LidarConeClusters>(
                    lp.topic, sensor_qos,
                    [this, idx](fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr msg) {
                        this->on_extra_lidar(msg, idx);
                    }));
        }

        if (motion_msg_type_ == "twist_stamped") {
            sub_motion_twist_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
                motion_topic_, sensor_qos,
                [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
                    this->on_motion_twist(msg);
                });
        } else if (motion_msg_type_ == "twist_with_covariance_stamped") {
            sub_motion_twist_cov_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
                motion_topic_, sensor_qos,
                [this](geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr msg) {
                    this->on_motion_twist_cov(msg);
                });
        } else {
            sub_motion_ = this->create_subscription<nav_msgs::msg::Odometry>(
                motion_topic_, sensor_qos,
                [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                    this->on_motion(msg);
                });
        }

#ifdef FUSION_HAS_TF2_ROS
        if (use_tf_extrinsics_) {
            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        }
#endif

        const double proc_period_s = 1.0 / std::max(1.0, process_rate_hz_);
        process_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(proc_period_s)),
            [this]() { this->process_once(); });

        const double diag_period_s = 1.0 / std::max(0.5, diag_rate_hz_);
        diag_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(diag_period_s)),
            [this]() { this->publish_diagnostics(); });

        RCLCPP_INFO(this->get_logger(), "fusion_core_cpp_node started (full parity C++20)");
    }

    ~FusionCoreComposableNode() override {
        for (auto& rec : frame_history_) {
            if (rec.snapshot_before) cpp_tracker_snapshot_destroy(rec.snapshot_before);
            if (rec.snapshot_after) cpp_tracker_snapshot_destroy(rec.snapshot_after);
        }
        frame_history_.clear();
        if (history_base_snapshot_) {
            cpp_tracker_snapshot_destroy(history_base_snapshot_);
            history_base_snapshot_ = nullptr;
        }
        if (tracker_handle_) {
            cpp_tracker_destroy(tracker_handle_);
            tracker_handle_ = nullptr;
        }
    }

private:
    struct CameraProfileState {
        std::string camera_id;
        std::string topic;
        std::string frame_id;
        std::string camera_info_topic;
        std::string depth_topic;
        bool use_tf_extrinsics{false};
        CameraIntrinsicsC intr{};
        ExtrinsicsC base_ext{};
        ExtrinsicsC dyn_ext{};
        std::deque<std::pair<double, fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr>> buffer;
        std::deque<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>> depth_buffer;
    };

    struct LidarProfileState {
        std::string lidar_id;
        std::string topic;
        std::string frame_id;
        double offset_x_m{0.0};
        double offset_y_m{0.0};
        double offset_z_m{0.0};
        double yaw_rad{0.0};
    };

    struct MotionSampleC {
        double vx{0.0};
        double vy{0.0};
        double vz{0.0};
        double wz{0.0};
    };

    struct FrameRecordC {
        double stamp{0.0};
        fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr lidar_msg;
        void* snapshot_before{nullptr};
        void* snapshot_after{nullptr};
    };

    void declare_and_load_parameters() {
        camera_topic_ = this->declare_parameter<std::string>("camera_topic", "/camera/cone_detections");
        lidar_topic_ = this->declare_parameter<std::string>("lidar_topic", "/lidar/cone_clusters");
        motion_topic_ = this->declare_parameter<std::string>("motion_topic", "/ego/odometry");
        motion_msg_type_ = this->declare_parameter<std::string>("motion_msg_type", "odometry");
        obs_out_topic_ = this->declare_parameter<std::string>("obs_out_topic", "/perception/cone_observations");
        track_out_topic_ = this->declare_parameter<std::string>("track_out_topic", "/perception/cone_tracks");
        diag_topic_ = this->declare_parameter<std::string>("diag_topic", "/perception/diagnostics");
        target_frame_ = this->declare_parameter<std::string>("target_frame", "base_link");

        enable_projection_semantics_ = this->declare_parameter("enable_projection_semantics", true);
        primary_intr_.fx = this->declare_parameter("camera_fx", 700.0);
        primary_intr_.fy = this->declare_parameter("camera_fy", 700.0);
        primary_intr_.cx = this->declare_parameter("camera_cx", 640.0);
        primary_intr_.cy = this->declare_parameter("camera_cy", 360.0);
        primary_intr_.width = std::max<int32_t>(1, this->declare_parameter("camera_width", 1280));
        primary_intr_.height = std::max<int32_t>(1, this->declare_parameter("camera_height", 720));
        primary_intr_.distortion_model = parse_distortion_model_c(
            this->declare_parameter<std::string>("camera_distortion_model", "plumb_bob"));
        primary_intr_.k1 = this->declare_parameter("camera_k1", 0.0);
        primary_intr_.k2 = this->declare_parameter("camera_k2", 0.0);
        primary_intr_.p1 = this->declare_parameter("camera_p1", 0.0);
        primary_intr_.p2 = this->declare_parameter("camera_p2", 0.0);
        primary_intr_.k3 = this->declare_parameter("camera_k3", 0.0);
        primary_intr_.k4 = this->declare_parameter("camera_k4", 0.0);

        const double tx = this->declare_parameter("lidar_to_camera_tx_m", 0.0);
        const double ty = this->declare_parameter("lidar_to_camera_ty_m", 0.0);
        const double tz = this->declare_parameter("lidar_to_camera_tz_m", 0.0);
        const double roll = this->declare_parameter("lidar_to_camera_roll_rad", -3.141592653589793);
        const double pitch = this->declare_parameter("lidar_to_camera_pitch_rad", -1.5707963267948966);
        const double yaw = this->declare_parameter("lidar_to_camera_yaw_rad", -1.5707963267948966);
        build_rpy_rotation(roll, pitch, yaw, primary_ext_.r);
        primary_ext_.t[0] = tx;
        primary_ext_.t[1] = ty;
        primary_ext_.t[2] = tz;

        use_tf_extrinsics_ = this->declare_parameter("use_tf_extrinsics", false);
        tf_extrinsics_lidar_frame_ = this->declare_parameter<std::string>("tf_extrinsics_lidar_frame", "");
        tf_extrinsics_camera_frame_ = this->declare_parameter<std::string>("tf_extrinsics_camera_frame", "");
        tf_lookup_timeout_ms_ = this->declare_parameter("tf_lookup_timeout_ms", 20.0);
        tf_extrinsics_warn_age_ms_ = this->declare_parameter("tf_extrinsics_warn_age_ms", 40.0);
        tf_extrinsics_critical_age_ms_ = this->declare_parameter("tf_extrinsics_critical_age_ms", 120.0);

        primary_ext_.sample_radius_scale = this->declare_parameter("projection_sample_radius_scale", 1.5);
        primary_ext_.min_depth_m = this->declare_parameter("projection_min_depth_m", 0.1);
        primary_ext_.max_depth_m = this->declare_parameter("projection_max_depth_m", 80.0);
        primary_ext_.min_semantic_weight = this->declare_parameter("projection_min_semantic_weight", 0.35);
        primary_ext_.no_match_semantic_weight = this->declare_parameter("projection_no_match_semantic_weight", 0.60);
        primary_ext_.soft_likelihood_enabled =
            this->declare_parameter("projection_soft_likelihood_enabled", true) ? 1 : 0;
        primary_ext_.soft_likelihood_sigma_scale = this->declare_parameter("projection_soft_likelihood_sigma_scale", 0.35);
        primary_ext_.soft_likelihood_min_value = this->declare_parameter("projection_soft_likelihood_min_value", 0.02);
        primary_ext_.soft_likelihood_outside_ratio = this->declare_parameter("projection_soft_likelihood_outside_ratio", 1.6);
        primary_ext_.enable_frustum_hull_sampling =
            this->declare_parameter("enable_frustum_hull_sampling", false) ? 1 : 0;
        primary_ext_.enable_triangular_cone_mask =
            this->declare_parameter("enable_triangular_cone_mask", true) ? 1 : 0;
        primary_ext_.cone_mask_apex_width_ratio = this->declare_parameter("cone_mask_apex_width_ratio", 0.18);

        projection_residual_warn_px_ = this->declare_parameter("projection_residual_warn_px", 5.0);
        projection_residual_critical_px_ = this->declare_parameter("projection_residual_critical_px", 12.0);
        enable_hungarian_early_fusion_ = this->declare_parameter("enable_hungarian_early_fusion", true);
        enable_chassis_attitude_compensation_ = this->declare_parameter("enable_chassis_attitude_compensation", true);
        aero_pitch_rad_per_mps2_ = this->declare_parameter("aero_pitch_rad_per_mps2", 0.0);
        aero_tz_m_per_mps2_ = this->declare_parameter("aero_tz_m_per_mps2", 0.0);

        enable_bidirectional_frustum_fusion_ = this->declare_parameter("enable_bidirectional_frustum_fusion", true);
        frustum_min_confidence_ = this->declare_parameter("frustum_min_confidence", 0.55);
        frustum_dedup_radius_m_ = this->declare_parameter("frustum_dedup_radius_m", 1.2);
        enable_ground_plane_depth_fusion_ = this->declare_parameter("enable_ground_plane_depth_fusion", true);
        ground_z_base_m_ = this->declare_parameter("ground_z_base_m", -0.22);
        ground_slope_rad_ = this->declare_parameter("ground_slope_rad", 0.0);

        enable_stereo_depth_fusion_ = this->declare_parameter("enable_stereo_depth_fusion", true);
        depth_topic_ = this->declare_parameter<std::string>("depth_topic", "");
        secondary_depth_topic_ = this->declare_parameter<std::string>("secondary_depth_topic", "");
        stereo_depth_scale_ = this->declare_parameter("stereo_depth_scale", 0.0);
        stereo_baseline_m_ = this->declare_parameter("stereo_baseline_m", 0.12);
        stereo_disparity_sigma_px_ = this->declare_parameter("stereo_disparity_sigma_px", 0.5);
        stereo_depth_max_age_ms_ = this->declare_parameter("stereo_depth_max_age_ms", 150.0);
        stereo_depth_min_m_ = this->declare_parameter("stereo_depth_min_m", 0.5);
        stereo_depth_max_m_ = this->declare_parameter("stereo_depth_max_m", 45.0);

        max_rewind_s_ = static_cast<double>(this->declare_parameter("max_rewind_ms", 150)) * 1e-3;
        max_replay_steps_ = this->declare_parameter("max_replay_steps", 8);
        frame_history_size_ = std::max<int32_t>(1, this->declare_parameter("frame_history_size", 128));
        max_camera_lidar_delta_s_ = static_cast<double>(this->declare_parameter("max_camera_lidar_delta_ms", 120)) * 1e-3;
        sync_offset_warn_ms_ = this->declare_parameter("sync_offset_warn_ms", 40.0);
        sync_offset_critical_ms_ = this->declare_parameter("sync_offset_critical_ms", 90.0);
        sync_stats_window_size_ = std::max<int32_t>(4, this->declare_parameter("sync_stats_window_size", 60));
        sync_jitter_warn_ms_ = this->declare_parameter("sync_jitter_warn_ms", 10.0);
        sync_jitter_critical_ms_ = this->declare_parameter("sync_jitter_critical_ms", 25.0);
        camera_retention_s_ = this->declare_parameter("camera_retention_s", 2.0);

        enable_camera_fallback_ = this->declare_parameter("enable_camera_fallback", true);
        camera_fallback_max_age_s_ = this->declare_parameter("camera_fallback_max_age_s", 0.25);
        camera_fallback_min_confidence_ = this->declare_parameter("camera_fallback_min_confidence", 0.25);
        camera_fallback_min_bbox_height_px_ = this->declare_parameter("camera_fallback_min_bbox_height_px", 8.0);
        camera_fallback_cone_height_m_ = this->declare_parameter("camera_fallback_cone_height_m", 0.30);
        camera_fallback_min_depth_m_ = this->declare_parameter("camera_fallback_min_depth_m", 2.0);
        camera_fallback_max_depth_m_ = this->declare_parameter("camera_fallback_max_depth_m", 35.0);
        camera_fallback_quality_scale_ = this->declare_parameter("camera_fallback_quality_scale", 0.45);
        camera_fallback_cov_xy_m2_ = this->declare_parameter("camera_fallback_cov_xy_m2", 4.0);
        camera_fallback_cov_z_m2_ = this->declare_parameter("camera_fallback_cov_z_m2", 8.0);

        queue_size_camera_ = std::max<int32_t>(1, this->declare_parameter("queue_size_camera", 16));
        queue_size_lidar_ = std::max<int32_t>(1, this->declare_parameter("queue_size_lidar", 16));
        queue_size_motion_ = std::max<int32_t>(1, this->declare_parameter("queue_size_motion", 64));
        drop_policy_ = this->declare_parameter<std::string>("drop_policy", "drop_oldest");
        blind_distance_target_speed_mps_ = this->declare_parameter("blind_distance_target_speed_mps", 16.0);
        motion_age_warn_ms_ = this->declare_parameter("motion_age_warn_ms", 80.0);
        motion_age_critical_ms_ = this->declare_parameter("motion_age_critical_ms", 180.0);
        motion_comp_error_gain_ = this->declare_parameter("motion_comp_error_gain", 0.6);

        process_rate_hz_ = this->declare_parameter("process_rate_hz", 30.0);
        diag_rate_hz_ = this->declare_parameter("diag_rate_hz", 2.0);

        enable_tracks_ = this->declare_parameter("enable_tracks", true);
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
        trk_cfg_.corridor_prior_weight = this->declare_parameter("track_corridor_prior_weight", 0.0);
        trk_cfg_.enable_global_map = this->declare_parameter("enable_global_map", false) ? 1 : 0;
        trk_cfg_.global_map_gate_radius_m = this->declare_parameter("global_map_gate_radius_m", 1.5);
        trk_cfg_.global_map_min_lap_distance_m = this->declare_parameter("global_map_min_lap_distance_m", 40.0);

        enable_lidar_stripe_semantics_ = this->declare_parameter("enable_lidar_stripe_semantics", true);
        require_camera_for_single_point_clusters_ =
            this->declare_parameter("require_camera_for_single_point_clusters", true);
        assume_uncounted_clusters_valid_ =
            this->declare_parameter("assume_uncounted_clusters_valid", true);
        min_uncorroborated_cluster_points_ =
            this->declare_parameter("min_uncorroborated_cluster_points", 2);

        enable_corridor_output_ = this->declare_parameter("enable_corridor_output", false);
        corridor_markers_topic_ = this->declare_parameter<std::string>("corridor_markers_topic", "/perception/corridor_markers");
        corridor_path_topic_ = this->declare_parameter<std::string>("corridor_path_topic", "/perception/corridor_path");
        corridor_horizon_m_ = this->declare_parameter("corridor_horizon_m", 20.0);
        corridor_step_m_ = this->declare_parameter("corridor_step_m", 1.0);

        enable_debug_viz_ = this->declare_parameter("enable_debug_viz", false);
        debug_markers_topic_ = this->declare_parameter<std::string>("debug_markers_topic", "/perception/debug_markers");
        debug_reprojection_topic_ = this->declare_parameter<std::string>("debug_reprojection_topic", "/perception/debug_reprojection");

        enable_online_calibration_ = this->declare_parameter("enable_online_calibration", false);
        online_calibration_window_frames_ = std::max<int32_t>(5, this->declare_parameter("online_calibration_window_frames", 30));
        online_calibration_min_pairs_ = std::max<int32_t>(3, this->declare_parameter("online_calibration_min_pairs", 12));
        online_calibration_apply_extrinsics_ = this->declare_parameter("online_calibration_apply_extrinsics", false);

        secondary_camera_topic_ = this->declare_parameter<std::string>("secondary_camera_topic", "");
        enable_camera_info_intrinsics_ = this->declare_parameter("enable_camera_info_intrinsics", true);
        camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
        secondary_camera_info_topic_ = this->declare_parameter<std::string>("secondary_camera_info_topic", "");
        camera_profiles_json_ = this->declare_parameter<std::string>("camera_profiles_json", "");
        secondary_camera_fx_ = this->declare_parameter("secondary_camera_fx", 0.0);
        secondary_camera_fy_ = this->declare_parameter("secondary_camera_fy", 0.0);
        secondary_camera_cx_ = this->declare_parameter("secondary_camera_cx", 0.0);
        secondary_camera_cy_ = this->declare_parameter("secondary_camera_cy", 0.0);
        secondary_tx_m_ = this->declare_parameter("secondary_lidar_to_camera_tx_m", 0.0);
        secondary_ty_m_ = this->declare_parameter("secondary_lidar_to_camera_ty_m", 0.0);
        secondary_tz_m_ = this->declare_parameter("secondary_lidar_to_camera_tz_m", 0.0);
        secondary_roll_rad_ = this->declare_parameter("secondary_lidar_to_camera_roll_rad", 0.0);
        secondary_pitch_rad_ = this->declare_parameter("secondary_lidar_to_camera_pitch_rad", 0.0);
        secondary_yaw_rad_ = this->declare_parameter("secondary_lidar_to_camera_yaw_rad", 0.0);

        lidar_topics_extra_ = this->declare_parameter<std::vector<std::string>>("lidar_topics_extra", std::vector<std::string>{""});
        lidar_profiles_json_ = this->declare_parameter<std::string>("lidar_profiles_json", "");

        health_degrade_scale_ = this->declare_parameter("health_degrade_scale", 1.5);
        quality_warn_threshold_ = this->declare_parameter("quality_warn_threshold", 0.4);
        quality_critical_threshold_ = this->declare_parameter("quality_critical_threshold", 0.2);

        sensor_qos_depth_ = this->declare_parameter("sensor_qos_depth", 10);
        output_qos_depth_ = this->declare_parameter("output_qos_depth", 20);
        diag_qos_depth_ = this->declare_parameter("diag_qos_depth", 10);
    }

    void build_registries() {
        camera_profiles_.clear();
        CameraProfileState primary;
        primary.camera_id = "primary";
        primary.topic = camera_topic_;
        primary.frame_id = tf_extrinsics_camera_frame_.empty() ? "camera_optical" : tf_extrinsics_camera_frame_;
        primary.camera_info_topic = camera_info_topic_;
        primary.depth_topic = depth_topic_;
        primary.use_tf_extrinsics = use_tf_extrinsics_;
        primary.intr = primary_intr_;
        primary.base_ext = primary_ext_;
        primary.dyn_ext = primary_ext_;
        camera_profiles_.push_back(std::move(primary));

        if (!secondary_camera_topic_.empty() && secondary_camera_topic_ != camera_topic_) {
            CameraProfileState sec;
            sec.camera_id = "secondary";
            sec.topic = secondary_camera_topic_;
            sec.frame_id = "secondary_camera_optical";
            sec.camera_info_topic = secondary_camera_info_topic_;
            sec.depth_topic = secondary_depth_topic_;
            sec.intr = primary_intr_;
            if (secondary_camera_fx_ > 0.0) sec.intr.fx = secondary_camera_fx_;
            if (secondary_camera_fy_ > 0.0) sec.intr.fy = secondary_camera_fy_;
            if (secondary_camera_cx_ > 0.0) sec.intr.cx = secondary_camera_cx_;
            if (secondary_camera_cy_ > 0.0) sec.intr.cy = secondary_camera_cy_;
            sec.base_ext = primary_ext_;
            build_rpy_rotation(secondary_roll_rad_, secondary_pitch_rad_, secondary_yaw_rad_, sec.base_ext.r);
            sec.base_ext.t[0] = secondary_tx_m_;
            sec.base_ext.t[1] = secondary_ty_m_;
            sec.base_ext.t[2] = secondary_tz_m_;
            sec.dyn_ext = sec.base_ext;
            camera_profiles_.push_back(std::move(sec));
        }

        if (!camera_profiles_json_.empty()) {
            const auto entries = parse_profile_json(camera_profiles_json_);
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                std::string cid = e.key;
                if (auto it = e.strings.find("camera_id"); it != e.strings.end() && !it->second.empty()) {
                    cid = it->second;
                }
                if (cid.empty()) cid = "cam_" + std::to_string(i);
                auto it_topic = e.strings.find("topic");
                if (it_topic == e.strings.end() || it_topic->second.empty()) continue;

                CameraProfileState prof;
                prof.camera_id = cid;
                prof.topic = it_topic->second;
                prof.frame_id = e.strings.count("frame_id") ? e.strings.at("frame_id") : cid;
                prof.camera_info_topic = e.strings.count("camera_info_topic") ? e.strings.at("camera_info_topic") : "";
                prof.depth_topic = e.strings.count("depth_topic") ? e.strings.at("depth_topic") : "";
                prof.use_tf_extrinsics = e.bools.count("use_tf_extrinsics") ? e.bools.at("use_tf_extrinsics") : false;
                prof.intr = primary_intr_;
                auto num = [&](const char* k, double def) -> double {
                    auto it = e.numbers.find(k);
                    return (it != e.numbers.end()) ? it->second : def;
                };
                prof.intr.fx = num("fx", prof.intr.fx);
                prof.intr.fy = num("fy", prof.intr.fy);
                prof.intr.cx = num("cx", prof.intr.cx);
                prof.intr.cy = num("cy", prof.intr.cy);
                prof.intr.width = std::max<int32_t>(1, static_cast<int32_t>(num("width", prof.intr.width)));
                prof.intr.height = std::max<int32_t>(1, static_cast<int32_t>(num("height", prof.intr.height)));
                prof.intr.k1 = num("k1", prof.intr.k1);
                prof.intr.k2 = num("k2", prof.intr.k2);
                prof.intr.p1 = num("p1", prof.intr.p1);
                prof.intr.p2 = num("p2", prof.intr.p2);
                prof.intr.k3 = num("k3", prof.intr.k3);
                prof.intr.k4 = num("k4", prof.intr.k4);
                if (e.strings.count("distortion_model")) {
                    prof.intr.distortion_model = parse_distortion_model_c(e.strings.at("distortion_model"));
                }

                prof.base_ext = primary_ext_;
                const double ex_tx = num("tx_m", num("tx", primary_ext_.t[0]));
                const double ex_ty = num("ty_m", num("ty", primary_ext_.t[1]));
                const double ex_tz = num("tz_m", num("tz", primary_ext_.t[2]));
                if (e.numbers.count("roll_rad") || e.numbers.count("pitch_rad") || e.numbers.count("yaw_rad") ||
                    e.numbers.count("roll") || e.numbers.count("pitch") || e.numbers.count("yaw")) {
                    const double r_v = num("roll_rad", num("roll", 0.0));
                    const double p_v = num("pitch_rad", num("pitch", 0.0));
                    const double y_v = num("yaw_rad", num("yaw", 0.0));
                    build_rpy_rotation(r_v, p_v, y_v, prof.base_ext.r);
                }
                prof.base_ext.t[0] = ex_tx;
                prof.base_ext.t[1] = ex_ty;
                prof.base_ext.t[2] = ex_tz;
                prof.dyn_ext = prof.base_ext;

                bool updated = false;
                for (auto& existing : camera_profiles_) {
                    if (existing.camera_id == cid) {
                        existing = std::move(prof);
                        updated = true;
                        break;
                    }
                }
                if (!updated) camera_profiles_.push_back(std::move(prof));
            }
        }

        lidar_profiles_.clear();
        lidar_profiles_.push_back({"primary", lidar_topic_, target_frame_.empty() ? "lidar" : target_frame_, 0.0, 0.0, 0.0, 0.0});
        for (size_t i = 0; i < lidar_topics_extra_.size(); ++i) {
            const std::string& t = lidar_topics_extra_[i];
            if (t.empty() || t == lidar_topic_) continue;
            bool dup = false;
            for (const auto& lp : lidar_profiles_) {
                if (lp.topic == t) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                lidar_profiles_.push_back({"lidar_extra_" + std::to_string(i), t, "lidar_extra_" + std::to_string(i), 0.0, 0.0, 0.0, 0.0});
            }
        }
        if (!lidar_profiles_json_.empty()) {
            const auto entries = parse_profile_json(lidar_profiles_json_);
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                std::string lid_id = e.key;
                if (auto it = e.strings.find("lidar_id"); it != e.strings.end() && !it->second.empty()) {
                    lid_id = it->second;
                }
                if (lid_id.empty()) lid_id = "lidar_" + std::to_string(i);
                auto it_topic = e.strings.find("topic");
                if (it_topic == e.strings.end() || it_topic->second.empty()) continue;

                auto num = [&](const char* k, double def) -> double {
                    auto it = e.numbers.find(k);
                    return (it != e.numbers.end()) ? it->second : def;
                };
                LidarProfileState lp;
                lp.lidar_id = lid_id;
                lp.topic = it_topic->second;
                lp.frame_id = e.strings.count("frame_id") ? e.strings.at("frame_id") : lid_id;
                lp.offset_x_m = num("offset_x_m", num("tx_m", 0.0));
                lp.offset_y_m = num("offset_y_m", num("ty_m", 0.0));
                lp.offset_z_m = num("offset_z_m", num("tz_m", 0.0));
                lp.yaw_rad = num("yaw_rad", num("yaw", 0.0));

                bool updated = false;
                for (auto& existing : lidar_profiles_) {
                    if (existing.lidar_id == lid_id || existing.topic == lp.topic) {
                        existing = std::move(lp);
                        updated = true;
                        break;
                    }
                }
                if (!updated) lidar_profiles_.push_back(std::move(lp));
            }
        }
    }

    template <typename DequeT, typename ItemT>
    void push_sorted(DequeT& buf, const ItemT& item, size_t max_len, int64_t& overflow_counter) {
        buf.push_back(item);
        std::stable_sort(buf.begin(), buf.end(),
                         [](const ItemT& a, const ItemT& b) { return a.first < b.first; });
        while (buf.size() > max_len) {
            if (drop_policy_ == "drop_oldest") {
                buf.pop_front();
            } else {
                buf.pop_back();
            }
            ++overflow_counter;
        }
    }

    void request_replay(double anchor_stamp) {
        pending_replay_ = true;
        ++cnt_replay_requests_;
        if (!has_replay_anchor_) {
            replay_anchor_stamp_ = anchor_stamp;
            has_replay_anchor_ = true;
        } else {
            replay_anchor_stamp_ = std::min(replay_anchor_stamp_, anchor_stamp);
        }
    }

    void on_camera(const fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& msg, size_t cam_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double stamp = stamp_to_sec(msg->header.stamp);
        if (stamp + max_rewind_s_ < last_processed_lidar_stamp_) {
            ++cnt_oosm_dropped_;
            return;
        }
        if (stamp < last_processed_lidar_stamp_) {
            ++cnt_oosm_within_budget_;
            request_replay(stamp);
        }
        if (cam_idx < camera_profiles_.size()) {
            push_sorted(camera_profiles_[cam_idx].buffer, std::make_pair(stamp, msg),
                        static_cast<size_t>(queue_size_camera_), cnt_camera_overflow_);
        }
    }

    void on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg, size_t cam_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_camera_info_intrinsics_ || cam_idx >= camera_profiles_.size()) return;
        double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
        if (msg->k[0] > 0.0 && msg->k[4] > 0.0) {
            fx = msg->k[0]; cx = msg->k[2]; fy = msg->k[4]; cy = msg->k[5];
        } else if (msg->p[0] > 0.0 && msg->p[5] > 0.0) {
            fx = msg->p[0]; cx = msg->p[2]; fy = msg->p[5]; cy = msg->p[6];
        }
        if (fx <= 0.0 || fy <= 0.0 || msg->width == 0 || msg->height == 0) return;

        auto& intr = camera_profiles_[cam_idx].intr;
        intr.fx = fx;
        intr.fy = fy;
        intr.cx = cx;
        intr.cy = cy;
        intr.width = static_cast<int32_t>(msg->width);
        intr.height = static_cast<int32_t>(msg->height);
        intr.distortion_model = parse_distortion_model_c(msg->distortion_model);
        if (intr.distortion_model == 1) {
            intr.k1 = (msg->d.size() > 0) ? msg->d[0] : 0.0;
            intr.k2 = (msg->d.size() > 1) ? msg->d[1] : 0.0;
            intr.p1 = 0.0;
            intr.p2 = 0.0;
            intr.k3 = (msg->d.size() > 2) ? msg->d[2] : 0.0;
            intr.k4 = (msg->d.size() > 3) ? msg->d[3] : 0.0;
        } else {
            intr.k1 = (msg->d.size() > 0) ? msg->d[0] : 0.0;
            intr.k2 = (msg->d.size() > 1) ? msg->d[1] : 0.0;
            intr.p1 = (msg->d.size() > 2) ? msg->d[2] : 0.0;
            intr.p2 = (msg->d.size() > 3) ? msg->d[3] : 0.0;
            intr.k3 = (msg->d.size() > 4) ? msg->d[4] : 0.0;
            intr.k4 = 0.0;
        }
        ++cnt_camera_info_updates_;
    }

    void on_depth(const sensor_msgs::msg::Image::ConstSharedPtr& msg, size_t cam_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_stereo_depth_fusion_ || cam_idx >= camera_profiles_.size()) return;
        const double stamp = stamp_to_sec(msg->header.stamp);
        auto& dbuf = camera_profiles_[cam_idx].depth_buffer;
        int64_t dummy_ov = 0;
        push_sorted(dbuf, std::make_pair(stamp, msg), static_cast<size_t>(std::max<int32_t>(4, std::min<int32_t>(16, queue_size_camera_))), dummy_ov);
        ++cnt_stereo_depth_frames_;
    }

    void on_lidar(const fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double stamp = stamp_to_sec(msg->header.stamp);
        if (stamp + max_rewind_s_ < last_processed_lidar_stamp_) {
            ++cnt_oosm_dropped_;
            return;
        }
        if (stamp < last_processed_lidar_stamp_) {
            ++cnt_oosm_within_budget_;
            request_replay(stamp);
        }
        push_sorted(lidar_buffer_, std::make_pair(stamp, msg),
                    static_cast<size_t>(queue_size_lidar_), cnt_lidar_overflow_);
    }

    void on_extra_lidar(const fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr& msg, size_t lid_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++cnt_extra_lidar_messages_;
        if (lid_idx >= lidar_profiles_.size()) return;
        const auto& prof = lidar_profiles_[lid_idx];
        if (std::abs(prof.offset_x_m) < 1e-9 && std::abs(prof.offset_y_m) < 1e-9 &&
            std::abs(prof.offset_z_m) < 1e-9 && std::abs(prof.yaw_rad) < 1e-9) {
            const double stamp = stamp_to_sec(msg->header.stamp);
            if (stamp + max_rewind_s_ < last_processed_lidar_stamp_) {
                ++cnt_oosm_dropped_;
                return;
            }
            if (stamp < last_processed_lidar_stamp_) {
                ++cnt_oosm_within_budget_;
                request_replay(stamp);
            }
            push_sorted(lidar_buffer_, std::make_pair(stamp, msg),
                        static_cast<size_t>(queue_size_lidar_), cnt_lidar_overflow_);
            return;
        }

        auto transformed = std::make_shared<fusion_interfaces::msg::LidarConeClusters>();
        transformed->header = msg->header;
        if (!target_frame_.empty()) transformed->header.frame_id = target_frame_;
        transformed->lidar_quality = msg->lidar_quality;
        transformed->clusters.reserve(msg->clusters.size());
        const double cy = std::cos(prof.yaw_rad);
        const double sy = std::sin(prof.yaw_rad);
        for (const auto& c : msg->clusters) {
            fusion_interfaces::msg::LidarConeCluster nc = c;
            const double x = c.centroid.x, y = c.centroid.y, z = c.centroid.z;
            nc.centroid.x = cy * x - sy * y + prof.offset_x_m;
            nc.centroid.y = sy * x + cy * y + prof.offset_y_m;
            nc.centroid.z = z + prof.offset_z_m;
            if (std::abs(prof.yaw_rad) >= 1e-9) {
                const double c00 = c.covariance[0], c01 = c.covariance[1], c02 = c.covariance[2];
                const double c10 = c.covariance[3], c11 = c.covariance[4], c12 = c.covariance[5];
                const double c20 = c.covariance[6], c21 = c.covariance[7], c22 = c.covariance[8];
                const double t00 = cy * c00 - sy * c10, t01 = cy * c01 - sy * c11;
                const double t10 = sy * c00 + cy * c10, t11 = sy * c01 + cy * c11;
                nc.covariance[0] = t00 * cy - t01 * sy;
                nc.covariance[1] = t00 * sy + t01 * cy;
                nc.covariance[2] = cy * c02 - sy * c12;
                nc.covariance[3] = t10 * cy - t11 * sy;
                nc.covariance[4] = t10 * sy + t11 * cy;
                nc.covariance[5] = sy * c02 + cy * c12;
                nc.covariance[6] = c20 * cy - c21 * sy;
                nc.covariance[7] = c20 * sy + c21 * cy;
                nc.covariance[8] = c22;
            }
            transformed->clusters.push_back(nc);
        }
        const double stamp = stamp_to_sec(transformed->header.stamp);
        if (stamp + max_rewind_s_ < last_processed_lidar_stamp_) {
            ++cnt_oosm_dropped_;
            return;
        }
        if (stamp < last_processed_lidar_stamp_) {
            ++cnt_oosm_within_budget_;
            request_replay(stamp);
        }
        push_sorted(lidar_buffer_, std::make_pair(stamp, fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr(transformed)),
                    static_cast<size_t>(queue_size_lidar_), cnt_lidar_overflow_);
    }

    void ingest_motion_sample(double stamp, double vx, double vy, double vz, double wz) {
        MotionSampleC sample{vx, vy, vz, wz};
        push_sorted(motion_buffer_, std::make_pair(stamp, sample),
                    static_cast<size_t>(queue_size_motion_), cnt_motion_overflow_);
        last_ego_speed_mps_ = std::sqrt(vx * vx + vy * vy + vz * vz);
    }

    void on_motion(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double stamp = stamp_to_sec(msg->header.stamp);
        const auto& lin = msg->twist.twist.linear;
        ingest_motion_sample(stamp, lin.x, lin.y, lin.z, msg->twist.twist.angular.z);
        const auto& q = msg->pose.pose.orientation;
        const double q_norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
        if (q_norm > 1e-6) {
            const double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
            const double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
            last_ego_roll_rad_ = std::atan2(sinr_cosp, cosr_cosp);
            const double sinp = clamp_val(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0);
            last_ego_pitch_rad_ = std::asin(sinp);
        }
    }

    void on_motion_twist(const geometry_msgs::msg::TwistStamped::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double stamp = stamp_to_sec(msg->header.stamp);
        const auto& lin = msg->twist.linear;
        ingest_motion_sample(stamp, lin.x, lin.y, lin.z, msg->twist.angular.z);
    }

    void on_motion_twist_cov(const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double stamp = stamp_to_sec(msg->header.stamp);
        const auto& lin = msg->twist.twist.linear;
        ingest_motion_sample(stamp, lin.x, lin.y, lin.z, msg->twist.twist.angular.z);
    }

    void update_motion_alignment(double reference_stamp) {
        if (motion_buffer_.empty()) {
            last_motion_age_ms_ = std::numeric_limits<double>::infinity();
            last_motion_comp_error_m_ = 0.0;
            ++cnt_motion_missing_for_cov_;
            return;
        }
        double best_dt = 1e18;
        MotionSampleC best_sample{};
        for (const auto& pair : motion_buffer_) {
            const double dt = std::abs(pair.first - reference_stamp);
            if (dt < best_dt) {
                best_dt = dt;
                best_sample = pair.second;
            }
        }
        const double age_ms = best_dt * 1000.0;
        last_motion_age_ms_ = age_ms;
        max_motion_age_ms_ = std::max(max_motion_age_ms_, age_ms);
        const double speed = std::sqrt(
            best_sample.vx * best_sample.vx + best_sample.vy * best_sample.vy + best_sample.vz * best_sample.vz);
        last_motion_comp_error_m_ = speed * age_ms * 1e-3;
        max_motion_comp_error_m_ = std::max(max_motion_comp_error_m_, last_motion_comp_error_m_);
    }

    EgoMotionC compute_ego_motion_step(double prev_stamp, double curr_stamp) const {
        EgoMotionC ego{};
        if (motion_buffer_.empty()) return ego;
        double best_dt = 1e18;
        MotionSampleC nearest_sample{};
        for (const auto& pair : motion_buffer_) {
            const double d = std::abs(pair.first - curr_stamp);
            if (d < best_dt) {
                best_dt = d;
                nearest_sample = pair.second;
            }
        }
        if (best_dt > 0.5) return ego;

        ego.has_motion = 1;
        ego.vx = nearest_sample.vx;
        ego.vy = nearest_sample.vy;
        ego.wz = nearest_sample.wz;
        const double dt = (prev_stamp > 0.0) ? std::max(0.0, curr_stamp - prev_stamp) : 0.0;
        ego.dt = dt;
        if (dt <= 1e-6 || dt > 0.5) return ego;

        struct SampleTw { double s, vx, vy, wz; };
        std::vector<SampleTw> in_win;
        for (const auto& p : motion_buffer_) {
            if (p.first >= (prev_stamp - 0.02) && p.first <= (curr_stamp + 0.02)) {
                in_win.push_back({
                    p.first,
                    p.second.vx,
                    p.second.vy,
                    p.second.wz
                });
            }
        }
        if (in_win.size() < 2) return ego;

        double dx_acc = 0.0, dy_acc = 0.0, dyaw_acc = 0.0;
        double t_cursor = prev_stamp;
        for (size_t i = 0; i < in_win.size(); ++i) {
            const double t_next = std::min(curr_stamp, (i + 1 < in_win.size()) ? in_win[i + 1].s : curr_stamp);
            const double sub_dt = std::max(0.0, t_next - t_cursor);
            if (sub_dt <= 0.0) continue;
            const double dth = in_win[i].wz * sub_dt;
            const double cy = std::cos(dyaw_acc + 0.5 * dth);
            const double sy = std::sin(dyaw_acc + 0.5 * dth);
            dx_acc += (in_win[i].vx * cy - in_win[i].vy * sy) * sub_dt;
            dy_acc += (in_win[i].vx * sy + in_win[i].vy * cy) * sub_dt;
            dyaw_acc += dth;
            t_cursor = t_next;
            if (t_cursor >= curr_stamp) break;
        }
        ego.has_disp = 1;
        ego.dx = dx_acc;
        ego.dy = dy_acc;
        ego.dyaw = dyaw_acc;
        return ego;
    }

    void transform_point_between_stamps(
        double& px, double& py, double /*pz*/, double from_stamp, double to_stamp
    ) const {
        const double dt = to_stamp - from_stamp;
        if (std::abs(dt) <= 1e-4 || std::abs(dt) > 0.25) return;
        const EgoMotionC step = compute_ego_motion_step(std::min(from_stamp, to_stamp), std::max(from_stamp, to_stamp));
        if (step.has_motion == 0) return;
        const double sign = (dt >= 0.0) ? 1.0 : -1.0;
        const double abs_dt = std::abs(dt);
        const double dtheta = sign * (step.has_disp != 0 ? step.dyaw : step.wz * abs_dt);
        double dx = 0.0, dy = 0.0;
        if (step.has_disp != 0 && dt >= 0.0) {
            dx = step.dx;
            dy = step.dy;
        } else {
            const double vx = sign * step.vx;
            const double vy = sign * step.vy;
            if (std::abs(dtheta) < 1e-5) {
                dx = vx * abs_dt;
                dy = vy * abs_dt;
            } else {
                const double s = std::sin(dtheta) / dtheta;
                const double c = (1.0 - std::cos(dtheta)) / dtheta;
                dx = (vx * s - vy * c) * abs_dt;
                dy = (vx * c + vy * s) * abs_dt;
            }
        }
        const double tx = px - dx;
        const double ty = py - dy;
        const double ct = std::cos(dtheta);
        const double st = std::sin(dtheta);
        px = ct * tx + st * ty;
        py = -st * tx + ct * ty;
    }

    void apply_aero_and_attitude(CameraProfileState& cam) const {
        cam.dyn_ext = cam.base_ext;
        const double pitch = enable_chassis_attitude_compensation_ ? last_ego_pitch_rad_ : 0.0;
        const double roll = enable_chassis_attitude_compensation_ ? last_ego_roll_rad_ : 0.0;
        if (std::abs(aero_pitch_rad_per_mps2_) <= 1e-12 &&
            std::abs(aero_tz_m_per_mps2_) <= 1e-12 &&
            std::abs(pitch) <= 1e-12 &&
            std::abs(roll) <= 1e-12) {
            return;
        }
        const double v2 = std::max(0.0, last_ego_speed_mps_) * std::max(0.0, last_ego_speed_mps_);
        const double dpitch = aero_pitch_rad_per_mps2_ * v2 + pitch;
        const double droll = roll;
        const double dtz = aero_tz_m_per_mps2_ * v2;
        const double cp = std::cos(dpitch), sp = std::sin(dpitch);
        const double cr = std::cos(droll), sr = std::sin(droll);
        const double r_pitch[9] = {1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp};
        const double r_roll[9] = {cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0};
        double r_dyn[9];
        mat_mul_3x3(r_roll, r_pitch, r_dyn);
        mat_mul_3x3(r_dyn, cam.base_ext.r, cam.dyn_ext.r);
        cam.dyn_ext.t[1] = cam.base_ext.t[1] + dtz;
    }

    void update_camera_lidar_sync(double cam_stamp, double lidar_stamp) {
        const double delta_ms = (cam_stamp - lidar_stamp) * 1000.0;
        const double abs_delta_ms = std::abs(delta_ms);
        last_camera_lidar_delta_ms_ = delta_ms;
        last_camera_lidar_abs_delta_ms_ = abs_delta_ms;
        max_camera_lidar_abs_delta_ms_ = std::max(max_camera_lidar_abs_delta_ms_, abs_delta_ms);
        ++camera_lidar_delta_samples_;
        sync_delta_window_ms_.push_back(delta_ms);
        while (sync_delta_window_ms_.size() > static_cast<size_t>(sync_stats_window_size_)) {
            sync_delta_window_ms_.pop_front();
        }
        if (sync_delta_window_ms_.empty()) return;
        double sum = 0.0;
        for (double v : sync_delta_window_ms_) sum += v;
        const double mean = sum / static_cast<double>(sync_delta_window_ms_.size());
        double var = 0.0;
        std::vector<double> abs_vals;
        abs_vals.reserve(sync_delta_window_ms_.size());
        for (double v : sync_delta_window_ms_) {
            var += (v - mean) * (v - mean);
            abs_vals.push_back(std::abs(v));
        }
        var /= static_cast<double>(sync_delta_window_ms_.size());
        std::sort(abs_vals.begin(), abs_vals.end());
        const size_t p95_idx = std::min(abs_vals.size() - 1, static_cast<size_t>(std::ceil(0.95 * abs_vals.size())) - 1);
        sync_delta_mean_ms_ = mean;
        sync_delta_std_ms_ = std::sqrt(std::max(0.0, var));
        sync_delta_abs_p95_ms_ = abs_vals[p95_idx];
    }

    std::vector<fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr> lookup_active_cameras(
        double lidar_stamp,
        fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& out_primary
    ) {
        std::vector<fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr> by_cam(camera_profiles_.size(), nullptr);
        out_primary.reset();
        bool primary_synced = false;
        bool any_buffered = false;
        bool any_matched = false;

        for (size_t i = 0; i < camera_profiles_.size(); ++i) {
            auto& buf = camera_profiles_[i].buffer;
            while (!buf.empty() && (lidar_stamp - buf.front().first) > camera_retention_s_) {
                buf.pop_front();
            }
            if (buf.empty()) continue;
            any_buffered = true;
            double best_dt = 1e18;
            double best_stamp = 0.0;
            fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr best_msg;
            for (const auto& pair : buf) {
                const double dt = std::abs(pair.first - lidar_stamp);
                if (dt < best_dt) {
                    best_dt = dt;
                    best_stamp = pair.first;
                    best_msg = pair.second;
                }
            }
            if (i == 0) {
                update_camera_lidar_sync(best_stamp, lidar_stamp);
                primary_synced = true;
            }
            if (best_dt <= max_camera_lidar_delta_s_) {
                by_cam[i] = best_msg;
                any_matched = true;
                if (i == 0) out_primary = best_msg;
            }
        }
        if (!primary_synced && any_matched) {
            for (const auto& m : by_cam) {
                if (m) {
                    update_camera_lidar_sync(stamp_to_sec(m->header.stamp), lidar_stamp);
                    break;
                }
            }
        }
        if (!any_matched && any_buffered) {
            ++cnt_camera_miss_;
        }
        if (!out_primary) {
            for (const auto& m : by_cam) {
                if (m) {
                    out_primary = m;
                    break;
                }
            }
        }
        return by_cam;
    }

    sensor_msgs::msg::Image::ConstSharedPtr nearest_depth_for_cam(double ref_stamp, size_t cam_idx) const {
        if (!enable_stereo_depth_fusion_ || cam_idx >= camera_profiles_.size()) return nullptr;
        const auto& dbuf = camera_profiles_[cam_idx].depth_buffer;
        if (dbuf.empty()) return nullptr;
        double best_dt = 1e18;
        sensor_msgs::msg::Image::ConstSharedPtr best_img;
        for (const auto& pair : dbuf) {
            const double dt = std::abs(pair.first - ref_stamp);
            if (dt < best_dt) {
                best_dt = dt;
                best_img = pair.second;
            }
        }
        if (best_dt > stereo_depth_max_age_ms_ * 1e-3) return nullptr;
        return best_img;
    }

    bool fill_stereo_depth_c(
        const sensor_msgs::msg::Image::ConstSharedPtr& img,
        StereoDepthImageC& out_sd
    ) const {
        if (!img || img->data.empty() || img->width == 0 || img->height == 0) return false;
        std::string enc = img->encoding;
        for (auto& ch : enc) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        int32_t mode = 0;
        double scale = stereo_depth_scale_;
        if (enc == "32fc1" || enc == "32fc1_meters") {
            mode = 1;
            if (scale <= 0.0) scale = 1.0;
        } else if (enc == "16uc1" || enc == "16uc1_mm" || enc == "16uc1_millimeters") {
            mode = 2;
            if (scale <= 0.0) scale = 0.001;
        } else if (enc == "32fc1_disparity" || enc == "disparity") {
            mode = 3;
            if (scale <= 0.0) scale = 1.0;
        } else {
            return false;
        }
        out_sd.width = static_cast<int32_t>(img->width);
        out_sd.height = static_cast<int32_t>(img->height);
        out_sd.encoding = mode;
        out_sd.data = img->data.data();
        out_sd.row_step_bytes = static_cast<int32_t>(img->step);
        out_sd.depth_scale = scale;
        out_sd.baseline_m = stereo_baseline_m_;
        out_sd.disparity_sigma_px = stereo_disparity_sigma_px_;
        out_sd.min_depth_m = stereo_depth_min_m_;
        out_sd.max_depth_m = stereo_depth_max_m_;
        return true;
    }

    void refresh_tf_extrinsics(
        const fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr& lidar_msg,
        const fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& cam_msg
    ) {
        if (!use_tf_extrinsics_) {
            last_extrinsics_source_ = "params";
            last_tf_lookup_ok_ = true;
            last_tf_lookup_error_.clear();
            last_tf_age_ms_ = 0.0;
            return;
        }
#ifdef FUSION_HAS_TF2_ROS
        if (!tf_buffer_) return;
        std::string lidar_frame = tf_extrinsics_lidar_frame_.empty() ? lidar_msg->header.frame_id : tf_extrinsics_lidar_frame_;
        std::string cam_frame = tf_extrinsics_camera_frame_;
        if (cam_frame.empty() && cam_msg) cam_frame = cam_msg->header.frame_id;
        if (lidar_frame.empty() || cam_frame.empty()) {
            ++cnt_tf_extrinsics_missing_frame_;
            last_tf_lookup_ok_ = false;
            last_tf_lookup_error_ = "missing lidar/camera frame for tf lookup";
            last_extrinsics_source_ = "params";
            return;
        }
        try {
            const tf2::TimePoint tp(std::chrono::nanoseconds(
                static_cast<int64_t>(lidar_msg->header.stamp.sec) * 1000000000LL +
                static_cast<int64_t>(lidar_msg->header.stamp.nanosec)));
            const auto tf_msg = tf_buffer_->lookupTransform(
                cam_frame, lidar_frame, tp, tf2::durationFromSec(std::max(0.0, tf_lookup_timeout_ms_ * 1e-3)));
            const auto& t = tf_msg.transform.translation;
            const auto& q = tf_msg.transform.rotation;
            quat_to_rotation_matrix(q.x, q.y, q.z, q.w, camera_profiles_[0].base_ext.r);
            camera_profiles_[0].base_ext.t[0] = t.x;
            camera_profiles_[0].base_ext.t[1] = t.y;
            camera_profiles_[0].base_ext.t[2] = t.z;
            ++cnt_tf_lookup_success_;
            last_tf_lookup_ok_ = true;
            last_tf_lookup_error_.clear();
            last_extrinsics_source_ = "tf";
            last_tf_lidar_frame_ = lidar_frame;
            last_tf_camera_frame_ = cam_frame;
            const double tf_s = stamp_to_sec(tf_msg.header.stamp);
            const double lid_s = stamp_to_sec(lidar_msg->header.stamp);
            last_tf_age_ms_ = (tf_s <= 0.0) ? 0.0 : std::max(0.0, (lid_s - tf_s) * 1000.0);
            if (last_tf_age_ms_ >= tf_extrinsics_warn_age_ms_) ++cnt_tf_extrinsics_stale_;

            for (size_t i = 1; i < camera_profiles_.size(); ++i) {
                if (!camera_profiles_[i].use_tf_extrinsics || camera_profiles_[i].frame_id.empty()) continue;
                try {
                    const auto aux_tf = tf_buffer_->lookupTransform(
                        camera_profiles_[i].frame_id, lidar_frame, tp,
                        tf2::durationFromSec(std::max(0.0, tf_lookup_timeout_ms_ * 1e-3)));
                    quat_to_rotation_matrix(
                        aux_tf.transform.rotation.x, aux_tf.transform.rotation.y,
                        aux_tf.transform.rotation.z, aux_tf.transform.rotation.w,
                        camera_profiles_[i].base_ext.r);
                    camera_profiles_[i].base_ext.t[0] = aux_tf.transform.translation.x;
                    camera_profiles_[i].base_ext.t[1] = aux_tf.transform.translation.y;
                    camera_profiles_[i].base_ext.t[2] = aux_tf.transform.translation.z;
                    ++cnt_tf_lookup_success_;
                } catch (const tf2::TransformException&) {
                    ++cnt_tf_lookup_fail_;
                }
            }
        } catch (const tf2::TransformException& ex) {
            ++cnt_tf_lookup_fail_;
            last_tf_lookup_ok_ = false;
            last_tf_lookup_error_ = ex.what();
            last_extrinsics_source_ = "params";
        }
#else
        (void)lidar_msg;
        (void)cam_msg;
#endif
    }

    int32_t compute_health_code(
        const fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& cam_msg,
        const fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr& lidar_msg
    ) const {
        const double lq = (lidar_msg && lidar_msg->lidar_quality > 0.0f) ? static_cast<double>(lidar_msg->lidar_quality) : 1.0;
        if (lq < quality_critical_threshold_) return kSensorHealthCritical;
        if (use_tf_extrinsics_ && !last_tf_lookup_ok_) return kSensorHealthDegraded;
        if (last_tf_age_ms_ >= tf_extrinsics_critical_age_ms_) return kSensorHealthCritical;
        if (last_tf_age_ms_ >= tf_extrinsics_warn_age_ms_) return kSensorHealthDegraded;
        if (sync_delta_std_ms_ >= sync_jitter_critical_ms_) return kSensorHealthCritical;
        if (sync_delta_std_ms_ >= sync_jitter_warn_ms_) return kSensorHealthDegraded;
        if (std::isfinite(last_motion_age_ms_) && last_motion_age_ms_ >= motion_age_critical_ms_) return kSensorHealthCritical;
        if (std::isfinite(last_motion_age_ms_) && last_motion_age_ms_ >= motion_age_warn_ms_) return kSensorHealthDegraded;
        if (last_camera_lidar_abs_delta_ms_ >= sync_offset_critical_ms_) return kSensorHealthCritical;
        if (last_projection_residual_mean_px_ >= projection_residual_critical_px_) return kSensorHealthCritical;
        if (!cam_msg) return kSensorHealthDegraded;
        const double cq = (cam_msg->image_quality > 0.0f) ? static_cast<double>(cam_msg->image_quality) : 1.0;
        if (cq < quality_warn_threshold_ || lq < quality_warn_threshold_ ||
            last_camera_lidar_abs_delta_ms_ >= sync_offset_warn_ms_ ||
            last_projection_residual_mean_px_ >= projection_residual_warn_px_) {
            return kSensorHealthDegraded;
        }
        return kSensorHealthOk;
    }

    double projection_residual_penalty(double res_mean_px) const {
        if (res_mean_px <= projection_residual_warn_px_) return 1.0;
        if (res_mean_px >= projection_residual_critical_px_) return 0.5;
        const double span = std::max(1e-6, projection_residual_critical_px_ - projection_residual_warn_px_);
        const double ratio = clamp_val((res_mean_px - projection_residual_warn_px_) / span, 0.0, 1.0);
        return clamp_val(1.0 - 0.5 * ratio, 0.5, 1.0);
    }

    std::array<double, 9> scaled_covariance(
        const std::array<double, 9>& cov_in,
        int32_t health,
        double proj_res_mean_px
    ) const {
        std::array<double, 9> cov = cov_in;
        bool all_zero = true;
        for (double v : cov) {
            if (std::abs(v) >= 1e-12) { all_zero = false; break; }
        }
        if (all_zero) {
            cov = {0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.5};
        }
        double scale = 1.0;
        if (health == kSensorHealthDegraded) scale = health_degrade_scale_;
        else if (health == kSensorHealthCritical) scale = health_degrade_scale_ * 2.0;

        if (proj_res_mean_px > projection_residual_warn_px_) {
            if (proj_res_mean_px >= projection_residual_critical_px_) scale *= 2.0;
            else {
                const double span = std::max(1e-6, projection_residual_critical_px_ - projection_residual_warn_px_);
                scale *= 1.0 + clamp_val((proj_res_mean_px - projection_residual_warn_px_) / span, 0.0, 1.0);
            }
        }
        if (last_camera_lidar_abs_delta_ms_ > sync_offset_warn_ms_) {
            if (last_camera_lidar_abs_delta_ms_ >= sync_offset_critical_ms_) scale *= 1.75;
            else {
                const double span = std::max(1e-6, sync_offset_critical_ms_ - sync_offset_warn_ms_);
                scale *= 1.0 + 0.75 * clamp_val((last_camera_lidar_abs_delta_ms_ - sync_offset_warn_ms_) / span, 0.0, 1.0);
            }
        }
        if (std::isfinite(last_motion_age_ms_) && last_motion_age_ms_ > motion_age_warn_ms_) {
            if (last_motion_age_ms_ >= motion_age_critical_ms_) scale *= 1.75;
            else {
                const double span = std::max(1e-6, motion_age_critical_ms_ - motion_age_warn_ms_);
                scale *= 1.0 + 0.75 * clamp_val((last_motion_age_ms_ - motion_age_warn_ms_) / span, 0.0, 1.0);
            }
        }
        if (last_motion_comp_error_m_ > 0.0) {
            scale *= 1.0 + std::max(0.0, motion_comp_error_gain_) * last_motion_comp_error_m_;
        }
        for (double& v : cov) v *= scale;
        return cov;
    }

    static std::vector<Detection2DC> to_det2d_list(
        const fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& cam_msg
    ) {
        std::vector<Detection2DC> out;
        if (!cam_msg) return out;
        out.reserve(cam_msg->detections.size());
        for (const auto& d : cam_msg->detections) {
            Detection2DC dc{};
            dc.xmin = d.xmin;
            dc.ymin = d.ymin;
            dc.xmax = d.xmax;
            dc.ymax = d.ymax;
            std::array<double, 4> p{};
            normalize_probs4_f(d.class_probs, p);
            for (int k = 0; k < 4; ++k) dc.class_probs[k] = p[k];
            dc.confidence = clamp_val(d.confidence, 0.0, 1.0);
            dc.polygon_count = 0;
            dc.is_big_orange = (p[3] > 0.35 && p[3] >= std::max({p[0], p[1], p[2]})) ? 1 : 0;
            out.push_back(dc);
        }
        return out;
    }

    void build_observations(
        const fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr& lidar_msg,
        const fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& cam_msg,
        const std::vector<fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr>& cam_by_id,
        const std::string& source,
        std::vector<fusion_interfaces::msg::ConeObservation>& out_obs,
        std::vector<Observation2DC>& out_trk_obs
    ) {
        out_obs.clear();
        out_trk_obs.clear();
        const int32_t health = compute_health_code(cam_msg, lidar_msg);
        last_health_ = health;
        const double lidar_stamp_s = stamp_to_sec(lidar_msg->header.stamp);
        const std::string out_frame = target_frame_.empty() ? lidar_msg->header.frame_id : target_frame_;

        // Sort clusters nearest-first (parity with Python FusionNode)
        std::vector<fusion_interfaces::msg::LidarConeCluster> sorted_clusters = lidar_msg->clusters;
        std::sort(sorted_clusters.begin(), sorted_clusters.end(),
                  [](const auto& a, const auto& b) {
                      return (a.centroid.x * a.centroid.x + a.centroid.y * a.centroid.y) <
                             (b.centroid.x * b.centroid.x + b.centroid.y * b.centroid.y);
                  });

        const size_t n_cls = sorted_clusters.size();
        std::vector<std::vector<Detection2DC>> dets_by_cam(camera_profiles_.size());
        std::vector<double> cam_stamps(camera_profiles_.size(), lidar_stamp_s);
        bool active_cam_present = false;
        for (size_t ci = 0; ci < camera_profiles_.size(); ++ci) {
            if (cam_by_id[ci]) {
                apply_aero_and_attitude(camera_profiles_[ci]);
                dets_by_cam[ci] = to_det2d_list(cam_by_id[ci]);
                cam_stamps[ci] = stamp_to_sec(cam_by_id[ci]->header.stamp);
                if (!dets_by_cam[ci].empty()) active_cam_present = true;
            }
        }
        if (cam_msg) active_cam_present = true;

        std::array<double, 4> fallback_probs{0.25, 0.25, 0.25, 0.25};
        double fallback_sem_w = 0.6;
        if (!enable_projection_semantics_ && cam_msg && !cam_msg->detections.empty()) {
            double acc[4]{0.0, 0.0, 0.0, 0.0};
            double wsum = 0.0;
            double conf_sum = 0.0;
            for (const auto& d : cam_msg->detections) {
                const double w = std::max(0.01, static_cast<double>(d.confidence));
                for (int k = 0; k < 4; ++k) acc[k] += d.class_probs[k] * w;
                wsum += w;
                conf_sum += d.confidence;
            }
            if (wsum > 1e-6) {
                for (int k = 0; k < 4; ++k) fallback_probs[k] = acc[k] / wsum;
            }
            fallback_sem_w = clamp_val(conf_sum / cam_msg->detections.size(), 0.3, 1.0);
        }

        // Multi-camera batch semantics
        std::vector<ClusterSemanticC> sem_fused(n_cls);
        std::vector<std::vector<uint8_t>> matched_flags_by_cam(camera_profiles_.size());
        if (enable_projection_semantics_ && n_cls > 0 && active_cam_present) {
            std::vector<std::vector<ClusterSemanticC>> per_cam_sem(camera_profiles_.size());
            std::vector<bool> cam_ran(camera_profiles_.size(), false);
            for (size_t ci = 0; ci < camera_profiles_.size(); ++ci) {
                if (!cam_by_id[ci] || dets_by_cam[ci].empty()) continue;
                std::vector<ClusterInputC> cls_in(n_cls);
                for (size_t k = 0; k < n_cls; ++k) {
                    const auto& c = sorted_clusters[k];
                    double px = c.centroid.x, py = c.centroid.y, pz = c.centroid.z;
                    transform_point_between_stamps(px, py, pz, lidar_stamp_s, cam_stamps[ci]);
                    cls_in[k] = {px, py, pz, c.covariance[0], c.covariance[4], c.covariance[8]};
                }
                per_cam_sem[ci].resize(n_cls);
                cpp_estimate_batch_semantics(
                    static_cast<int32_t>(n_cls), cls_in.data(),
                    static_cast<int32_t>(dets_by_cam[ci].size()), dets_by_cam[ci].data(),
                    &camera_profiles_[ci].intr, &camera_profiles_[ci].dyn_ext,
                    fallback_probs.data(), per_cam_sem[ci].data());
                cam_ran[ci] = true;
                matched_flags_by_cam[ci].assign(dets_by_cam[ci].size(), 0);
                for (const auto& s : per_cam_sem[ci]) {
                    if (s.matched_detection_idx >= 0 &&
                        static_cast<size_t>(s.matched_detection_idx) < matched_flags_by_cam[ci].size()) {
                        matched_flags_by_cam[ci][static_cast<size_t>(s.matched_detection_idx)] = 1;
                    }
                }
            }

            for (size_t k = 0; k < n_cls; ++k) {
                double acc[4]{0.0, 0.0, 0.0, 0.0};
                double w_tot = 0.0;
                int32_t proj_s = 0, match_s = 0, res_s = 0, best_idx = -1;
                double w_sum = 0.0, best_conf = 0.0, res_sum = 0.0, res_max = 0.0, best_sem_w = 0.0;
                int matched_cams = 0;
                ClusterSemanticC best_unmatched{};
                int32_t best_unmatched_proj = -1;

                for (size_t ci = 0; ci < camera_profiles_.size(); ++ci) {
                    if (!cam_ran[ci]) continue;
                    const auto& s = per_cam_sem[ci][k];
                    if (s.matched_detection_idx >= 0) {
                        ++matched_cams;
                        const double w = std::max(1e-4, s.semantic_weight * std::max(0.1, s.best_confidence));
                        for (int m = 0; m < 4; ++m) acc[m] += w * s.class_probs[m];
                        w_tot += w;
                        proj_s += s.projected_samples;
                        match_s += s.matched_samples;
                        w_sum += s.matched_weight_sum;
                        if (s.best_confidence > best_conf) {
                            best_conf = s.best_confidence;
                            best_idx = s.matched_detection_idx;
                        }
                        res_s += s.residual_samples;
                        res_sum += s.residual_sum_px;
                        res_max = std::max(res_max, s.residual_max_px);
                        best_sem_w = std::max(best_sem_w, s.semantic_weight);
                    } else if (s.projected_samples > best_unmatched_proj) {
                        best_unmatched_proj = s.projected_samples;
                        best_unmatched = s;
                    }
                }
                if (matched_cams == 0) {
                    sem_fused[k] = (best_unmatched_proj >= 0) ? best_unmatched : ClusterSemanticC{};
                    if (best_unmatched_proj < 0) {
                        for (int m = 0; m < 4; ++m) sem_fused[k].class_probs[m] = fallback_probs[m];
                        sem_fused[k].semantic_weight = camera_profiles_[0].dyn_ext.no_match_semantic_weight;
                        sem_fused[k].matched_detection_idx = -1;
                    }
                } else {
                    auto& out = sem_fused[k];
                    for (int m = 0; m < 4; ++m) out.class_probs[m] = acc[m] / std::max(1e-9, w_tot);
                    out.semantic_weight = clamp_val(best_sem_w, 0.0, 1.0);
                    out.projected_samples = proj_s;
                    out.matched_samples = match_s;
                    out.matched_weight_sum = w_sum;
                    out.best_confidence = best_conf;
                    out.residual_samples = res_s;
                    out.residual_sum_px = res_sum;
                    out.residual_max_px = res_max;
                    out.matched_detection_idx = best_idx;
                }
            }
        }

        double frame_res_sum = 0.0;
        int32_t frame_res_samples = 0;
        std::vector<ClusterInputC> existing_clusters;
        existing_clusters.reserve(n_cls);

        for (size_t k = 0; k < n_cls; ++k) {
            const auto& c = sorted_clusters[k];
            const int32_t pt_cnt = (c.point_count == 0)
                                       ? (assume_uncounted_clusters_valid_ ? min_uncorroborated_cluster_points_ : 0)
                                       : static_cast<int32_t>(c.point_count);

            std::array<double, 4> c_probs = fallback_probs;
            double sem_w = fallback_sem_w;
            double cluster_res_mean = 0.0;
            int32_t matched_idx = -1;

            if (enable_projection_semantics_ && active_cam_present) {
                const auto& s = sem_fused[k];
                for (int m = 0; m < 4; ++m) c_probs[m] = s.class_probs[m];
                sem_w = s.semantic_weight;
                matched_idx = s.matched_detection_idx;

                cnt_proj_samples_ += s.projected_samples;
                cnt_matched_samples_ += s.matched_samples;
                if (s.projected_samples > 0 && s.matched_samples == 0) ++cnt_clusters_no_match_;
                if (s.residual_samples > 0) {
                    cluster_res_mean = s.residual_sum_px / static_cast<double>(s.residual_samples);
                    frame_res_sum += s.residual_sum_px;
                    frame_res_samples += s.residual_samples;
                    proj_res_sum_total_ += s.residual_sum_px;
                    proj_res_samples_total_ += s.residual_samples;
                    proj_res_max_px_ = std::max(proj_res_max_px_, s.residual_max_px);
                    sem_w *= projection_residual_penalty(cluster_res_mean);
                    if (cluster_res_mean >= projection_residual_warn_px_) ++cnt_high_res_clusters_;
                }
            }

            if (matched_idx < 0 && enable_lidar_stripe_semantics_) {
                const float max_sp = std::max({c.stripe_class_probs[0], c.stripe_class_probs[1],
                                               c.stripe_class_probs[2], c.stripe_class_probs[3]});
                if (max_sp > 0.35f) {
                    for (int m = 0; m < 4; ++m) c_probs[m] = c.stripe_class_probs[m];
                }
            }

            if (require_camera_for_single_point_clusters_ &&
                pt_cnt < min_uncorroborated_cluster_points_ && matched_idx < 0) {
                continue;
            }
            existing_clusters.push_back({c.centroid.x, c.centroid.y, c.centroid.z, 0.05, 0.05, 0.05});

            fusion_interfaces::msg::ConeObservation obs_msg;
            obs_msg.header.stamp = lidar_msg->header.stamp;
            obs_msg.header.frame_id = out_frame;
            obs_msg.position = c.centroid;
            const auto sc_cov = scaled_covariance(c.covariance, health, cluster_res_mean);
            for (int m = 0; m < 9; ++m) obs_msg.covariance[m] = sc_cov[m];
            for (int m = 0; m < 4; ++m) obs_msg.class_probs[m] = static_cast<float>(c_probs[m]);
            const double geom_q = (c.geometry_quality > 0.0f) ? static_cast<double>(c.geometry_quality) : 1.0;
            double q = geom_q * sem_w;
            if (health == kSensorHealthDegraded) q *= 0.8;
            else if (health == kSensorHealthCritical) q *= 0.6;
            obs_msg.quality_score = static_cast<float>(clamp_val(q, 0.0, 1.0));
            obs_msg.sensor_health = static_cast<uint8_t>(health);
            obs_msg.source = source;
            out_obs.push_back(obs_msg);

            if (enable_tracks_) {
                out_trk_obs.push_back({
                    obs_msg.position.x, obs_msg.position.y, obs_msg.position.z,
                    std::max(1e-6, obs_msg.covariance[0]),
                    0.5 * (obs_msg.covariance[1] + obs_msg.covariance[3]),
                    std::max(1e-6, obs_msg.covariance[4]),
                    std::max(1e-6, obs_msg.covariance[8]),
                    {c_probs[0], c_probs[1], c_probs[2], c_probs[3]},
                    obs_msg.quality_score
                });
            }
        }

        // Bidirectional Camera -> 3D Frustum + Stereo Depth Recovery
        if (enable_bidirectional_frustum_fusion_ && enable_projection_semantics_ && active_cam_present) {
            FrustumRecoveryConfigC fcfg{};
            fcfg.cone_height_m = camera_fallback_cone_height_m_;
            fcfg.big_orange_height_m = 0.505;
            fcfg.min_confidence = frustum_min_confidence_;
            fcfg.min_bbox_height_px = camera_fallback_min_bbox_height_px_;
            fcfg.min_depth_m = camera_fallback_min_depth_m_;
            fcfg.max_depth_m = camera_fallback_max_depth_m_;
            fcfg.dedup_radius_m = frustum_dedup_radius_m_;
            fcfg.range_std_fraction = 0.14;
            fcfg.bearing_std_fraction = 0.025;
            fcfg.enable_ground_plane_depth_fusion = enable_ground_plane_depth_fusion_ ? 1 : 0;
            fcfg.ground_z_base_m = ground_z_base_m_;
            fcfg.ground_slope_rad = ground_slope_rad_;
            fcfg.enable_stereo_depth_fusion = enable_stereo_depth_fusion_ ? 1 : 0;
            fcfg.enable_triangular_cone_mask = primary_ext_.enable_triangular_cone_mask;
            fcfg.cone_mask_apex_width_ratio = primary_ext_.cone_mask_apex_width_ratio;

            std::vector<ClusterInputC> occupied = existing_clusters;
            for (size_t ci = 0; ci < camera_profiles_.size(); ++ci) {
                if (!cam_by_id[ci] || dets_by_cam[ci].empty()) continue;
                StereoDepthImageC sd{};
                auto depth_img = nearest_depth_for_cam(cam_stamps[ci], ci);
                const bool has_sd = fill_stereo_depth_c(depth_img, sd);
                std::vector<RecoveredConeC> rec_buf(64);
                const int32_t n_rec = cpp_recover_unmatched_detections(
                    static_cast<int32_t>(dets_by_cam[ci].size()), dets_by_cam[ci].data(),
                    matched_flags_by_cam[ci].empty() ? nullptr : matched_flags_by_cam[ci].data(),
                    static_cast<int32_t>(occupied.size()), occupied.empty() ? nullptr : occupied.data(),
                    &camera_profiles_[ci].intr, &camera_profiles_[ci].dyn_ext, &fcfg,
                    has_sd ? &sd : nullptr, rec_buf.data(), 64);
                const double img_q = std::max(0.0, static_cast<double>(cam_by_id[ci]->image_quality));
                for (int32_t r = 0; r < n_rec; ++r) {
                    const auto& rc = rec_buf[static_cast<size_t>(r)];
                    double px = rc.x, py = rc.y, pz = rc.z;
                    occupied.push_back({px, py, pz, 0.05, 0.05, 0.05});
                    transform_point_between_stamps(px, py, pz, cam_stamps[ci], lidar_stamp_s);
                    std::array<double, 9> rcov{};
                    for (int m = 0; m < 9; ++m) rcov[m] = rc.covariance[m];
                    const auto sc_cov = scaled_covariance(rcov, health, 0.0);

                    fusion_interfaces::msg::ConeObservation obs_msg;
                    obs_msg.header.stamp = lidar_msg->header.stamp;
                    obs_msg.header.frame_id = out_frame;
                    obs_msg.position.x = px;
                    obs_msg.position.y = py;
                    obs_msg.position.z = pz;
                    for (int m = 0; m < 9; ++m) obs_msg.covariance[m] = sc_cov[m];
                    for (int m = 0; m < 4; ++m) obs_msg.class_probs[m] = static_cast<float>(rc.class_probs[m]);
                    obs_msg.quality_score = static_cast<float>(clamp_val(
                        rc.confidence * img_q * (rc.used_stereo ? 0.82 : 0.70), 0.1, 1.0));
                    obs_msg.sensor_health = static_cast<uint8_t>(health);
                    obs_msg.source = rc.used_stereo ? (source + "_stereo_recovered") : (source + "_frustum_recovered");
                    out_obs.push_back(obs_msg);
                    ++cnt_frustum_recovered_;
                    if (rc.used_stereo) ++cnt_stereo_fused_;
                    if (enable_tracks_) {
                        out_trk_obs.push_back({
                            px, py, pz,
                            std::max(1e-6, obs_msg.covariance[0]),
                            0.5 * (obs_msg.covariance[1] + obs_msg.covariance[3]),
                            std::max(1e-6, obs_msg.covariance[4]),
                            std::max(1e-6, obs_msg.covariance[8]),
                            {rc.class_probs[0], rc.class_probs[1], rc.class_probs[2], rc.class_probs[3]},
                            obs_msg.quality_score
                        });
                    }
                }
            }
        }

        if (!existing_clusters.empty() && enable_projection_semantics_ && !camera_profiles_.empty()) {
            const auto& cam0 = camera_profiles_[0];
            int32_t forward_cnt = 0, in_front_cnt = 0, behind_cnt = 0, in_fov_cnt = 0;
            const double* R = cam0.dyn_ext.r;
            const double* t = cam0.dyn_ext.t;
            for (const auto& c : existing_clusters) {
                if (c.cx > 0.5 || c.cz > 0.5) ++forward_cnt;
                const double xc = R[0] * c.cx + R[1] * c.cy + R[2] * c.cz + t[0];
                const double yc = R[3] * c.cx + R[4] * c.cy + R[5] * c.cz + t[1];
                const double zc = R[6] * c.cx + R[7] * c.cy + R[8] * c.cz + t[2];
                if (zc > cam0.dyn_ext.min_depth_m) {
                    ++in_front_cnt;
                    const double xn = xc / zc, yn = yc / zc;
                    double xd = xn, yd = yn;
                    if (cam0.intr.distortion_model == 1) {
                        const double r = std::hypot(xn, yn);
                        if (r > 1e-12) {
                            const double th = std::atan(r);
                            const double t2 = th * th, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
                            const double sc = th * (1.0 + cam0.intr.k1 * t2 + cam0.intr.k2 * t4 + cam0.intr.k3 * t6 + cam0.intr.k4 * t8) / r;
                            xd = xn * sc; yd = yn * sc;
                        }
                    }
                    const double pu = cam0.intr.fx * xd + cam0.intr.cx;
                    const double pv = cam0.intr.fy * yd + cam0.intr.cy;
                    if (pu >= 0.0 && pu < static_cast<double>(cam0.intr.width) &&
                        pv >= 0.0 && pv < static_cast<double>(cam0.intr.height)) {
                        ++in_fov_cnt;
                    }
                } else {
                    ++behind_cnt;
                }
            }
            if (forward_cnt > 0 && in_front_cnt == 0 && behind_cnt > 0) {
                last_extrinsics_sanity_ = "error_points_behind_camera_check_optical_frame";
            } else if (in_front_cnt > 0 && in_fov_cnt == 0) {
                last_extrinsics_sanity_ = "warn_points_outside_image_fov";
            } else {
                last_extrinsics_sanity_ = "ok";
            }
            if (last_extrinsics_sanity_ != "ok") {
                ++extrinsics_sanity_warn_streak_;
                if (extrinsics_sanity_warn_streak_ == 5 || extrinsics_sanity_warn_streak_ == 50 ||
                    (extrinsics_sanity_warn_streak_ > 50 && extrinsics_sanity_warn_streak_ % 100 == 0)) {
                    RCLCPP_WARN(this->get_logger(),
                                "Extrinsics/frame sanity check reported '%s' (streak=%d). Check optical frame or TF extrinsics.",
                                last_extrinsics_sanity_.c_str(), extrinsics_sanity_warn_streak_);
                }
            } else {
                extrinsics_sanity_warn_streak_ = 0;
            }
        }

        if (enable_camera_fallback_ && out_obs.empty() && cam_msg && !cam_msg->detections.empty()) {
            build_camera_fallback_observations(cam_msg, lidar_stamp_s, source + "_camera_fallback", out_obs, out_trk_obs);
        }

        if (frame_res_samples > 0) {
            last_projection_residual_mean_px_ = frame_res_sum / static_cast<double>(frame_res_samples);
        } else if (!enable_projection_semantics_ || !active_cam_present) {
            last_projection_residual_mean_px_ = 0.0;
        }

        if (enable_debug_viz_ && cam_msg && !dets_by_cam[0].empty() && !existing_clusters.empty()) {
            publish_reprojection_debug(existing_clusters, dets_by_cam[0], lidar_msg->header, lidar_stamp_s, cam_stamps[0]);
        }

        if (enable_online_calibration_ && (source == "live" || source == "oosm_live") &&
            cam_msg && !dets_by_cam[0].empty() && !existing_clusters.empty()) {
            for (const auto& c : existing_clusters) {
                double px = c.cx, py = c.cy, pz = c.cz;
                transform_point_between_stamps(px, py, pz, lidar_stamp_s, cam_stamps[0]);
                // Pair with nearest detection
                double best_d2 = 1e9;
                int best_di = -1;
                for (size_t di = 0; di < dets_by_cam[0].size(); ++di) {
                    const auto& d = dets_by_cam[0][di];
                    const double du = 0.5 * (d.xmin + d.xmax);
                    const double dv = 0.5 * (d.ymin + d.ymax);
                    if (du >= 0.0 && dv >= 0.0) {
                        // Simple nearest in image after current projection
                        const double* R = camera_profiles_[0].base_ext.r;
                        const double* t = camera_profiles_[0].base_ext.t;
                        const double xc = R[0]*px + R[1]*py + R[2]*pz + t[0];
                        const double yc = R[3]*px + R[4]*py + R[5]*pz + t[1];
                        const double zc = R[6]*px + R[7]*py + R[8]*pz + t[2];
                        if (zc > 0.5) {
                            const double pu = camera_profiles_[0].intr.fx * (xc / zc) + camera_profiles_[0].intr.cx;
                            const double pv = camera_profiles_[0].intr.fy * (yc / zc) + camera_profiles_[0].intr.cy;
                            const double d2 = (pu - du) * (pu - du) + (pv - dv) * (pv - dv);
                            if (d2 < best_d2 && d2 < 10000.0) {
                                best_d2 = d2;
                                best_di = static_cast<int>(di);
                            }
                        }
                    }
                }
                if (best_di >= 0) {
                    const auto& d = dets_by_cam[0][static_cast<size_t>(best_di)];
                    calib_pairs_window_.push_back({
                        px, py, pz,
                        0.5 * (d.xmin + d.xmax), 0.5 * (d.ymin + d.ymax),
                        d.xmax - d.xmin, d.ymax - d.ymin
                    });
                }
            }
            const size_t max_pairs_buf = static_cast<size_t>(online_calibration_window_frames_) * 16;
            while (calib_pairs_window_.size() > max_pairs_buf) calib_pairs_window_.pop_front();
            ++online_calib_frames_since_run_;
            if (online_calib_frames_since_run_ >= online_calibration_window_frames_ &&
                static_cast<int32_t>(calib_pairs_window_.size()) >= online_calibration_min_pairs_) {
                online_calib_frames_since_run_ = 0;
                std::vector<OnlineCalibPairC> pv(calib_pairs_window_.begin(), calib_pairs_window_.end());
                OnlineCalibResultC cres{};
                ++cnt_online_calib_runs_;
                if (cpp_online_calibrate_extrinsics(
                        static_cast<int32_t>(pv.size()), pv.data(),
                        &camera_profiles_[0].intr, &camera_profiles_[0].base_ext,
                        online_calibration_min_pairs_, &cres) != 0) {
                    if (online_calibration_apply_extrinsics_) {
                        for (int m = 0; m < 9; ++m) camera_profiles_[0].base_ext.r[m] = cres.refined_r[m];
                        ++cnt_online_calib_applied_;
                    }
                }
            }
        }
    }

    void build_camera_fallback_observations(
        const fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr& cam_msg,
        double stamp_sec,
        const std::string& source,
        std::vector<fusion_interfaces::msg::ConeObservation>& out_obs,
        std::vector<Observation2DC>& out_trk_obs
    ) {
        if (!cam_msg || cam_msg->detections.empty()) return;
        const std::string out_frame = target_frame_.empty() ? cam_msg->header.frame_id : target_frame_;
        const int32_t health = kSensorHealthDegraded;
        last_health_ = std::max(last_health_, health);

        apply_aero_and_attitude(camera_profiles_[0]);
        auto dets = to_det2d_list(cam_msg);
        StereoDepthImageC sd{};
        auto depth_img = nearest_depth_for_cam(stamp_sec, 0);
        const bool has_sd = fill_stereo_depth_c(depth_img, sd);

        FrustumRecoveryConfigC fcfg{};
        fcfg.cone_height_m = camera_fallback_cone_height_m_;
        fcfg.big_orange_height_m = 0.505;
        fcfg.min_confidence = camera_fallback_min_confidence_;
        fcfg.min_bbox_height_px = camera_fallback_min_bbox_height_px_;
        fcfg.min_depth_m = camera_fallback_min_depth_m_;
        fcfg.max_depth_m = camera_fallback_max_depth_m_;
        fcfg.dedup_radius_m = 0.5;
        fcfg.range_std_fraction = 0.14;
        fcfg.bearing_std_fraction = 0.025;
        fcfg.enable_ground_plane_depth_fusion = enable_ground_plane_depth_fusion_ ? 1 : 0;
        fcfg.ground_z_base_m = ground_z_base_m_;
        fcfg.ground_slope_rad = ground_slope_rad_;
        fcfg.enable_stereo_depth_fusion = enable_stereo_depth_fusion_ ? 1 : 0;
        fcfg.enable_triangular_cone_mask = primary_ext_.enable_triangular_cone_mask;
        fcfg.cone_mask_apex_width_ratio = primary_ext_.cone_mask_apex_width_ratio;

        std::vector<RecoveredConeC> rec_buf(64);
        const int32_t n_rec = cpp_recover_unmatched_detections(
            static_cast<int32_t>(dets.size()), dets.data(), nullptr,
            0, nullptr, &camera_profiles_[0].intr, &camera_profiles_[0].dyn_ext, &fcfg,
            has_sd ? &sd : nullptr, rec_buf.data(), 64);

        const double img_q = std::max(0.0, static_cast<double>(cam_msg->image_quality));
        for (int32_t i = 0; i < n_rec; ++i) {
            const auto& rc = rec_buf[static_cast<size_t>(i)];
            std::array<double, 9> base_cov{};
            if (rc.used_stereo) {
                for (int m = 0; m < 9; ++m) base_cov[m] = rc.covariance[m];
            } else {
                base_cov = {
                    std::max(rc.covariance[0], camera_fallback_cov_xy_m2_), rc.covariance[1], 0.0,
                    rc.covariance[3], std::max(rc.covariance[4], camera_fallback_cov_xy_m2_), 0.0,
                    0.0, 0.0, std::max(rc.covariance[8], camera_fallback_cov_z_m2_)
                };
            }
            const auto sc_cov = scaled_covariance(base_cov, health, 0.0);
            fusion_interfaces::msg::ConeObservation obs_msg;
            obs_msg.header.stamp = cam_msg->header.stamp;
            obs_msg.header.frame_id = out_frame;
            obs_msg.position.x = rc.x;
            obs_msg.position.y = rc.y;
            obs_msg.position.z = rc.z;
            for (int m = 0; m < 9; ++m) obs_msg.covariance[m] = sc_cov[m];
            for (int m = 0; m < 4; ++m) obs_msg.class_probs[m] = static_cast<float>(rc.class_probs[m]);
            const double q_scale = std::max(camera_fallback_quality_scale_, rc.used_stereo ? 0.75 : 0.0);
            obs_msg.quality_score = static_cast<float>(clamp_val(rc.confidence * img_q * q_scale, 0.0, 1.0));
            obs_msg.sensor_health = static_cast<uint8_t>(health);
            obs_msg.source = rc.used_stereo ? (source + "_stereo") : source;
            out_obs.push_back(obs_msg);
            if (rc.used_stereo) ++cnt_stereo_fused_;
            if (enable_tracks_) {
                out_trk_obs.push_back({
                    rc.x, rc.y, rc.z,
                    std::max(1e-6, obs_msg.covariance[0]),
                    0.5 * (obs_msg.covariance[1] + obs_msg.covariance[3]),
                    std::max(1e-6, obs_msg.covariance[4]),
                    std::max(1e-6, obs_msg.covariance[8]),
                    {rc.class_probs[0], rc.class_probs[1], rc.class_probs[2], rc.class_probs[3]},
                    obs_msg.quality_score
                });
            }
        }
    }

    void apply_tracker_step(
        const std::vector<Observation2DC>& trk_obs,
        const std_msgs::msg::Header& header,
        double stamp
    ) {
        const EgoMotionC ego = compute_ego_motion_step(last_tracker_stamp_, stamp);
        last_tracker_stamp_ = stamp;

        std::array<TrackEstimateC, 256> active_tracks{};
        std::array<TrackEstimateC, 64> lost_tracks{};
        TrackerStepStatsC stats{};
        const int32_t n_tracks = cpp_tracker_step(
            tracker_handle_, stamp, &ego,
            static_cast<int32_t>(trk_obs.size()), trk_obs.empty() ? nullptr : trk_obs.data(),
            active_tracks.data(), 256, lost_tracks.data(), 64, &stats);

        cnt_tracks_created_ += stats.tracks_created;
        cnt_tracks_deleted_ += stats.tracks_deleted;
        cnt_assoc_candidates_ += stats.assoc_candidates;
        cnt_assoc_matches_ += stats.assoc_matches;
        cnt_assoc_rejects_ += stats.assoc_rejects;
        last_track_count_ = n_tracks;

        const std::string out_frame = target_frame_.empty() ? header.frame_id : target_frame_;
        const auto now_msg = this->now();
        auto pub_tr = [&](const TrackEstimateC& tr, uint8_t status, const builtin_interfaces::msg::Time& st) {
            fusion_interfaces::msg::ConeTrack msg;
            msg.header.stamp = st;
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
            for (int m = 0; m < 4; ++m) msg.class_probs[m] = static_cast<float>(tr.class_probs[m]);
            msg.track_age = static_cast<float>(std::max(0.0, tr.last_stamp - tr.first_stamp));
            msg.track_status = status;
            msg.quality_score = static_cast<float>(tr.quality);
            pub_tracks_->publish(msg);
        };

        const int32_t n_lost = std::min<int32_t>(stats.n_lost_tracks, 64);
        for (int32_t i = 0; i < n_lost; ++i) {
            pub_tr(lost_tracks[static_cast<size_t>(i)], fusion_interfaces::msg::ConeTrack::TRACK_LOST, now_msg);
        }
        for (int32_t i = 0; i < n_tracks; ++i) {
            const auto& tr = active_tracks[static_cast<size_t>(i)];
            pub_tr(tr, static_cast<uint8_t>(tr.status), header.stamp);
        }

        CorridorEstimateC corridor{};
        bool has_corridor = false;
        if (enable_corridor_output_) {
            if (cpp_tracker_get_corridor(tracker_handle_, corridor_horizon_m_, corridor_step_m_, &ego, &corridor) != 0) {
                has_corridor = true;
                publish_corridor(corridor, header);
            }
        }
        if (enable_debug_viz_) {
            publish_debug_scene_markers(active_tracks.data(), n_tracks, has_corridor ? &corridor : nullptr, header);
        }
    }

    void publish_corridor(const CorridorEstimateC& cor, const std_msgs::msg::Header& header) {
        const std::string out_frame = target_frame_.empty() ? header.frame_id : target_frame_;
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = header.stamp;
        path_msg.header.frame_id = out_frame;
        path_msg.poses.reserve(static_cast<size_t>(cor.n_waypoints));
        for (int32_t i = 0; i < cor.n_waypoints; ++i) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path_msg.header;
            ps.pose.position.x = cor.wp_x[i];
            ps.pose.position.y = cor.wp_y[i];
            const double hy = 0.5 * cor.wp_heading[i];
            ps.pose.orientation.z = std::sin(hy);
            ps.pose.orientation.w = std::cos(hy);
            path_msg.poses.push_back(ps);
        }
        pub_corridor_path_->publish(path_msg);

        visualization_msgs::msg::MarkerArray ma;
        auto add_strip = [&](const char* ns, int32_t id, const double* xs, const double* ys, double z,
                             float r, float g, float b, float a, double w) {
            if (cor.n_waypoints <= 0) return;
            visualization_msgs::msg::Marker m;
            m.header = path_msg.header;
            m.ns = ns;
            m.id = id;
            m.type = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.orientation.w = 1.0;
            m.scale.x = w;
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
            m.points.reserve(static_cast<size_t>(cor.n_waypoints));
            for (int32_t i = 0; i < cor.n_waypoints; ++i) {
                geometry_msgs::msg::Point pt;
                pt.x = xs[i]; pt.y = ys[i]; pt.z = z;
                m.points.push_back(pt);
            }
            ma.markers.push_back(std::move(m));
        };
        add_strip("corridor_left", 0, cor.left_x, cor.left_y, 0.05, 0.12f, 0.47f, 0.96f, 0.92f, 0.08);
        add_strip("corridor_right", 1, cor.right_x, cor.right_y, 0.05, 0.98f, 0.85f, 0.10f, 0.92f, 0.08);
        add_strip("corridor_centerline", 2, cor.wp_x, cor.wp_y, 0.08, 0.15f, 0.95f, 0.35f, 0.95f, 0.10);
        pub_corridor_markers_->publish(ma);
        ++cnt_corridor_published_;
    }

    void publish_debug_scene_markers(
        const TrackEstimateC* tracks,
        int32_t n_tracks,
        const CorridorEstimateC* cor,
        const std_msgs::msg::Header& header
    ) {
        static constexpr float kRgba[4][4] = {
            {0.12f, 0.47f, 0.96f, 0.92f},
            {0.98f, 0.85f, 0.10f, 0.92f},
            {0.98f, 0.45f, 0.08f, 0.92f},
            {1.00f, 0.22f, 0.05f, 0.96f}
        };
        static constexpr const char* kNames[4] = {"BLUE", "YELLOW", "ORANGE", "BIG_ORANGE"};
        const std::string out_frame = target_frame_.empty() ? header.frame_id : target_frame_;
        visualization_msgs::msg::MarkerArray ma;
        for (int32_t i = 0; i < n_tracks; ++i) {
            const auto& tr = tracks[i];
            int cls = 0;
            for (int k = 1; k < 4; ++k) {
                if (tr.class_probs[k] > tr.class_probs[cls]) cls = k;
            }
            const bool is_big = (cls == 3);
            const double cone_h = is_big ? 0.505 : 0.325;
            const double cone_d = is_big ? 0.285 : 0.228;

            visualization_msgs::msg::Marker cyl;
            cyl.header.stamp = header.stamp;
            cyl.header.frame_id = out_frame;
            cyl.ns = "cones_3d";
            cyl.id = tr.track_id;
            cyl.type = visualization_msgs::msg::Marker::CYLINDER;
            cyl.action = visualization_msgs::msg::Marker::ADD;
            cyl.pose.position.x = tr.state[0];
            cyl.pose.position.y = tr.state[1];
            cyl.pose.position.z = tr.z;
            cyl.pose.orientation.w = 1.0;
            cyl.scale.x = cone_d; cyl.scale.y = cone_d; cyl.scale.z = cone_h;
            cyl.color.r = kRgba[cls][0]; cyl.color.g = kRgba[cls][1]; cyl.color.b = kRgba[cls][2];
            cyl.color.a = static_cast<float>(clamp_val(kRgba[cls][3] * tr.quality, 0.35, 1.0));
            ma.markers.push_back(std::move(cyl));

            double maj_d = 0.1, min_d = 0.1, ell_yaw = 0.0;
            compute_cov_ellipse_2sigma(tr.covariance[0], tr.covariance[1], tr.covariance[5], maj_d, min_d, ell_yaw);
            visualization_msgs::msg::Marker ell;
            ell.header.stamp = header.stamp;
            ell.header.frame_id = out_frame;
            ell.ns = "cones_cov_2sigma";
            ell.id = tr.track_id;
            ell.type = visualization_msgs::msg::Marker::CYLINDER;
            ell.action = visualization_msgs::msg::Marker::ADD;
            ell.pose.position.x = tr.state[0];
            ell.pose.position.y = tr.state[1];
            ell.pose.position.z = tr.z - 0.5 * cone_h + 0.02;
            ell.pose.orientation.z = std::sin(0.5 * ell_yaw);
            ell.pose.orientation.w = std::cos(0.5 * ell_yaw);
            ell.scale.x = std::max(0.08, maj_d);
            ell.scale.y = std::max(0.08, min_d);
            ell.scale.z = 0.015;
            ell.color.r = kRgba[cls][0]; ell.color.g = kRgba[cls][1]; ell.color.b = kRgba[cls][2]; ell.color.a = 0.32f;
            ma.markers.push_back(std::move(ell));

            visualization_msgs::msg::Marker txt;
            txt.header.stamp = header.stamp;
            txt.header.frame_id = out_frame;
            txt.ns = "cones_labels";
            txt.id = tr.track_id;
            txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            txt.action = visualization_msgs::msg::Marker::ADD;
            txt.pose.position.x = tr.state[0];
            txt.pose.position.y = tr.state[1];
            txt.pose.position.z = tr.z + 0.5 * cone_h + 0.18;
            txt.pose.orientation.w = 1.0;
            txt.scale.x = 0.16; txt.scale.y = 0.16; txt.scale.z = 0.16;
            txt.color.r = 1.0f; txt.color.g = 1.0f; txt.color.b = 1.0f; txt.color.a = 0.95f;
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "#%d %s (%.0f%%)", tr.track_id, kNames[cls], tr.class_probs[cls] * 100.0);
            txt.text = lbl;
            ma.markers.push_back(std::move(txt));
        }
        if (cor != nullptr && cor->n_waypoints > 0) {
            auto add_c = [&](int32_t id, const double* xs, const double* ys, float r, float g, float b, double w) {
                visualization_msgs::msg::Marker m;
                m.header.stamp = header.stamp;
                m.header.frame_id = out_frame;
                m.ns = "corridor_boundaries";
                m.id = id;
                m.type = visualization_msgs::msg::Marker::LINE_STRIP;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose.orientation.w = 1.0;
                m.scale.x = w;
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9f;
                for (int32_t i = 0; i < cor->n_waypoints; ++i) {
                    geometry_msgs::msg::Point pt;
                    pt.x = xs[i]; pt.y = ys[i]; pt.z = -0.35;
                    m.points.push_back(pt);
                }
                ma.markers.push_back(std::move(m));
            };
            add_c(1, cor->left_x, cor->left_y, 0.15f, 0.52f, 0.98f, 0.06);
            add_c(2, cor->right_x, cor->right_y, 0.98f, 0.85f, 0.12f, 0.06);
            add_c(3, cor->wp_x, cor->wp_y, 0.12f, 0.92f, 0.45f, 0.08);
        }
        pub_debug_markers_->publish(ma);
        ++cnt_debug_markers_published_;
    }

    void publish_reprojection_debug(
        const std::vector<ClusterInputC>& clusters,
        const std::vector<Detection2DC>& dets,
        const std_msgs::msg::Header& header,
        double lidar_stamp_s,
        double cam_stamp_s
    ) {
        static constexpr const char* kNames[4] = {"BLUE", "YELLOW", "ORANGE", "BIG_ORANGE"};
        std::ostringstream oss;
        oss << "{\"frame_id\":\"" << header.frame_id << "\",\"annotations\":[";
        bool first = true;
        const auto& intr = camera_profiles_[0].intr;
        const auto& ext = camera_profiles_[0].dyn_ext;
        for (size_t i = 0; i < clusters.size(); ++i) {
            double px = clusters[i].cx, py = clusters[i].cy, pz = clusters[i].cz;
            transform_point_between_stamps(px, py, pz, lidar_stamp_s, cam_stamp_s);
            const double xc = ext.r[0] * px + ext.r[1] * py + ext.r[2] * pz + ext.t[0];
            const double yc = ext.r[3] * px + ext.r[4] * py + ext.r[5] * pz + ext.t[1];
            const double zc = ext.r[6] * px + ext.r[7] * py + ext.r[8] * pz + ext.t[2];
            if (zc <= 0.5 || zc >= 60.0) continue;
            const double pu = intr.fx * (xc / zc) + intr.cx;
            const double pv = intr.fy * (yc / zc) + intr.cy;
            if (pu < 0.0 || pu >= intr.width || pv < 0.0 || pv >= intr.height) continue;

            int best_j = -1;
            double best_res = 100.0;
            std::string best_lbl = "UNMATCHED";
            for (size_t j = 0; j < dets.size(); ++j) {
                const double du = 0.5 * (dets[j].xmin + dets[j].xmax);
                const double dv = 0.5 * (dets[j].ymin + dets[j].ymax);
                const double d = std::hypot(pu - du, pv - dv);
                if (d < best_res) {
                    best_res = d;
                    best_j = static_cast<int>(j);
                    int cls = 0;
                    for (int k = 1; k < 4; ++k) {
                        if (dets[j].class_probs[k] > dets[j].class_probs[cls]) cls = k;
                    }
                    best_lbl = kNames[cls];
                }
            }
            if (!first) oss << ",";
            first = false;
            oss << "{\"cluster_index\":" << i
                << ",\"u_proj\":" << fmt_double(pu, 2)
                << ",\"v_proj\":" << fmt_double(pv, 2)
                << ",\"depth_m\":" << fmt_double(zc, 3)
                << ",\"matched_detection_index\":" << best_j
                << ",\"residual_px\":" << fmt_double(best_j >= 0 ? best_res : -1.0, 2)
                << ",\"class_label\":\"" << best_lbl << "\"}";
        }
        oss << "]}";
        std_msgs::msg::String msg;
        msg.data = oss.str();
        pub_debug_reprojection_->publish(msg);
    }

    void upsert_frame_record(FrameRecordC&& rec) {
        bool replaced = false;
        for (size_t i = 0; i < frame_history_.size(); ++i) {
            if (std::abs(frame_history_[i].stamp - rec.stamp) <= 1e-9) {
                if (frame_history_[i].snapshot_before) cpp_tracker_snapshot_destroy(frame_history_[i].snapshot_before);
                if (frame_history_[i].snapshot_after) cpp_tracker_snapshot_destroy(frame_history_[i].snapshot_after);
                frame_history_[i] = std::move(rec);
                replaced = true;
                break;
            }
            if (frame_history_[i].stamp > rec.stamp) {
                frame_history_.insert(frame_history_.begin() + static_cast<ptrdiff_t>(i), std::move(rec));
                replaced = true;
                break;
            }
        }
        if (!replaced) frame_history_.push_back(std::move(rec));
        while (frame_history_.size() > static_cast<size_t>(frame_history_size_)) {
            auto& front = frame_history_.front();
            if (front.snapshot_before) cpp_tracker_snapshot_destroy(front.snapshot_before);
            if (front.snapshot_after) {
                if (history_base_snapshot_) cpp_tracker_snapshot_destroy(history_base_snapshot_);
                history_base_snapshot_ = front.snapshot_after;
                front.snapshot_after = nullptr;
            }
            frame_history_.pop_front();
        }
    }

    void* resolve_replay_base_snapshot(const FrameRecordC& first_frame) const {
        if (first_frame.snapshot_before) return first_frame.snapshot_before;
        for (auto it = frame_history_.rbegin(); it != frame_history_.rend(); ++it) {
            if (it->stamp < first_frame.stamp && it->snapshot_after) return it->snapshot_after;
        }
        return history_base_snapshot_;
    }

    void run_replay_if_needed() {
        const auto t0 = std::chrono::steady_clock::now();
        if (!pending_replay_) {
            record_stage_timing("replay_total", 0.0);
            return;
        }
        pending_replay_ = false;
        if (!enable_tracks_ || !has_replay_anchor_) {
            has_replay_anchor_ = false;
            record_stage_timing("replay_total", elapsed_ms(t0));
            return;
        }
        const double anchor = replay_anchor_stamp_;
        has_replay_anchor_ = false;

        std::vector<size_t> replay_indices;
        for (size_t i = 0; i < frame_history_.size(); ++i) {
            if (frame_history_[i].stamp >= anchor) replay_indices.push_back(i);
        }
        if (replay_indices.empty()) {
            ++cnt_replay_no_frames_;
            record_stage_timing("replay_total", elapsed_ms(t0));
            return;
        }
        if (static_cast<int32_t>(replay_indices.size()) > max_replay_steps_) {
            ++cnt_replay_budget_exceeded_;
            record_stage_timing("replay_total", elapsed_ms(t0));
            return;
        }
        void* base_snap = resolve_replay_base_snapshot(frame_history_[replay_indices.front()]);
        if (!base_snap) {
            ++cnt_replay_no_base_snapshot_;
            record_stage_timing("replay_total", elapsed_ms(t0));
            return;
        }
        cpp_tracker_snapshot_restore(tracker_handle_, base_snap);
        last_tracker_stamp_ = cpp_tracker_get_last_stamp(tracker_handle_);

        int32_t replayed = 0;
        for (size_t idx : replay_indices) {
            auto& f = frame_history_[idx];
            update_motion_alignment(f.stamp);
            fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr cam_msg;
            auto cam_by_id = lookup_active_cameras(f.stamp, cam_msg);
            refresh_tf_extrinsics(f.lidar_msg, cam_msg);

            std::vector<fusion_interfaces::msg::ConeObservation> obs_msgs;
            std::vector<Observation2DC> trk_obs;
            build_observations(f.lidar_msg, cam_msg, cam_by_id, "replay", obs_msgs, trk_obs);
            for (const auto& om : obs_msgs) pub_obs_->publish(om);

            void* sb = cpp_tracker_snapshot_create(tracker_handle_);
            apply_tracker_step(trk_obs, f.lidar_msg->header, f.stamp);
            void* sa = cpp_tracker_snapshot_create(tracker_handle_);
            if (f.snapshot_before) cpp_tracker_snapshot_destroy(f.snapshot_before);
            if (f.snapshot_after) cpp_tracker_snapshot_destroy(f.snapshot_after);
            f.snapshot_before = sb;
            f.snapshot_after = sa;
            ++replayed;
        }

        ++cnt_replay_completed_;
        cnt_replay_frames_ += replayed;
        last_processed_lidar_stamp_ = std::max(last_processed_lidar_stamp_, frame_history_[replay_indices.back()].stamp);
        record_stage_timing("replay_total", elapsed_ms(t0));
    }

    void process_camera_fallback_without_lidar() {
        const double now_s = static_cast<double>(this->now().nanoseconds()) * 1e-9;
        auto& buf = camera_profiles_[0].buffer;
        while (!buf.empty() && (now_s - buf.front().first) > camera_retention_s_) {
            buf.pop_front();
        }
        if (buf.empty()) {
            ++cnt_cam_fb_no_cam_;
            return;
        }
        double best_dt = 1e18, cam_stamp = 0.0;
        fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr cam_msg;
        for (const auto& pair : buf) {
            const double d = std::abs(pair.first - now_s);
            if (d < best_dt) {
                best_dt = d;
                cam_stamp = pair.first;
                cam_msg = pair.second;
            }
        }
        if ((now_s - cam_stamp) > camera_fallback_max_age_s_) {
            ++cnt_cam_fb_stale_;
            return;
        }
        if (cam_stamp <= last_camera_fallback_stamp_ + 1e-9) {
            ++cnt_cam_fb_dup_;
            return;
        }
        update_motion_alignment(cam_stamp);
        std::vector<fusion_interfaces::msg::ConeObservation> fb_obs;
        std::vector<Observation2DC> fb_trk;
        build_camera_fallback_observations(cam_msg, cam_stamp, "camera_fallback_live", fb_obs, fb_trk);
        if (fb_obs.empty()) return;
        for (const auto& om : fb_obs) pub_obs_->publish(om);
        last_obs_count_ = static_cast<int32_t>(fb_obs.size());
        last_camera_fallback_stamp_ = cam_stamp;
        ++cnt_cam_fb_frames_;
        cnt_cam_fb_obs_ += static_cast<int64_t>(fb_obs.size());
        if (enable_tracks_) {
            apply_tracker_step(fb_trk, cam_msg->header, cam_stamp);
        }
    }

    void process_once() {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto cycle_t0 = std::chrono::steady_clock::now();
        if (lidar_buffer_.empty()) {
            if (enable_camera_fallback_) process_camera_fallback_without_lidar();
            if (pending_replay_) run_replay_if_needed();
            record_stage_timing("cycle_total", elapsed_ms(cycle_t0));
            return;
        }

        const double lidar_stamp = lidar_buffer_.front().first;
        const auto lidar_msg = lidar_buffer_.front().second;
        lidar_buffer_.pop_front();

        update_motion_alignment(lidar_stamp);
        if (lidar_stamp + max_rewind_s_ < last_processed_lidar_stamp_) {
            ++cnt_oosm_dropped_;
            record_stage_timing("cycle_total", elapsed_ms(cycle_t0));
            return;
        }

        const auto cam_t0 = std::chrono::steady_clock::now();
        fusion_interfaces::msg::CameraConeDetections::ConstSharedPtr cam_msg;
        const auto cam_by_id = lookup_active_cameras(lidar_stamp, cam_msg);
        record_stage_timing("camera_lookup", elapsed_ms(cam_t0));

        const auto ext_t0 = std::chrono::steady_clock::now();
        refresh_tf_extrinsics(lidar_msg, cam_msg);
        record_stage_timing("extrinsics_lookup", elapsed_ms(ext_t0));

        const bool is_oosm_lidar = (lidar_stamp < last_processed_lidar_stamp_);
        const std::string source = is_oosm_lidar ? "oosm_live" : "live";

        const auto build_t0 = std::chrono::steady_clock::now();
        std::vector<fusion_interfaces::msg::ConeObservation> obs_msgs;
        std::vector<Observation2DC> trk_obs;
        build_observations(lidar_msg, cam_msg, cam_by_id, source, obs_msgs, trk_obs);
        record_stage_timing("build_observations", elapsed_ms(build_t0));

        const auto pub_t0 = std::chrono::steady_clock::now();
        for (const auto& om : obs_msgs) pub_obs_->publish(om);
        record_stage_timing("publish_observations", elapsed_ms(pub_t0));
        last_obs_count_ = static_cast<int32_t>(obs_msgs.size());

        if (enable_tracks_) {
            if (is_oosm_lidar) {
                record_stage_timing("tracker_step", 0.0);
                upsert_frame_record({lidar_stamp, lidar_msg, nullptr, nullptr});
                request_replay(lidar_stamp);
            } else {
                const auto trk_t0 = std::chrono::steady_clock::now();
                void* sb = cpp_tracker_snapshot_create(tracker_handle_);
                apply_tracker_step(trk_obs, lidar_msg->header, lidar_stamp);
                record_stage_timing("tracker_step", elapsed_ms(trk_t0));
                void* sa = cpp_tracker_snapshot_create(tracker_handle_);
                upsert_frame_record({lidar_stamp, lidar_msg, sb, sa});
            }
        } else {
            record_stage_timing("tracker_step", 0.0);
            upsert_frame_record({lidar_stamp, lidar_msg, nullptr, nullptr});
        }

        if (!is_oosm_lidar) {
            last_processed_lidar_stamp_ = std::max(last_processed_lidar_stamp_, lidar_stamp);
        }

        const double now_s = static_cast<double>(this->now().nanoseconds()) * 1e-9;
        last_latency_ms_ = std::max(0.0, (now_s - lidar_stamp) * 1000.0);
        max_latency_ms_ = std::max(max_latency_ms_, last_latency_ms_);
        last_blind_distance_current_m_ = last_ego_speed_mps_ * last_latency_ms_ * 1e-3;
        last_blind_distance_target_m_ = blind_distance_target_speed_mps_ * last_latency_ms_ * 1e-3;
        max_blind_distance_target_m_ = std::max(max_blind_distance_target_m_, last_blind_distance_target_m_);
        ++cycles_completed_;

        if (pending_replay_) run_replay_if_needed();
        record_stage_timing("cycle_total", elapsed_ms(cycle_t0));
    }

    static double elapsed_ms(const std::chrono::steady_clock::time_point& t0) {
        return std::max(0.0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }

    void record_stage_timing(const std::string& stage, double ms) {
        last_stage_ms_[stage] = ms;
        max_stage_ms_[stage] = std::max(max_stage_ms_[stage], ms);
    }

    void publish_diagnostics() {
        std::lock_guard<std::mutex> lock(mutex_);
        diagnostic_msgs::msg::DiagnosticArray msg;
        msg.header.stamp = this->now();
        diagnostic_msgs::msg::DiagnosticStatus st;
        st.name = "fusion_core_cpp/runtime";
        st.hardware_id = "fusion_core_cpp";
        st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        st.message = "ok";

        if (last_health_ == kSensorHealthCritical) {
            st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            st.message = "critical sensor health";
        } else if (last_health_ == kSensorHealthDegraded ||
                   cnt_camera_overflow_ > 0 || cnt_lidar_overflow_ > 0 ||
                   cnt_oosm_dropped_ > 0 || cnt_replay_budget_exceeded_ > 0 ||
                   last_camera_lidar_abs_delta_ms_ >= sync_offset_warn_ms_ ||
                   sync_delta_std_ms_ >= sync_jitter_warn_ms_ ||
                   last_projection_residual_mean_px_ >= projection_residual_warn_px_ ||
                   (std::isfinite(last_motion_age_ms_) && last_motion_age_ms_ >= motion_age_warn_ms_) ||
                   (last_extrinsics_sanity_ != "ok" && last_extrinsics_sanity_ != "no_clusters") ||
                   (use_tf_extrinsics_ && (!last_tf_lookup_ok_ || last_tf_age_ms_ >= tf_extrinsics_warn_age_ms_))) {
            st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            st.message = (last_extrinsics_sanity_ != "ok" && last_extrinsics_sanity_ != "no_clusters")
                             ? ("extrinsics sanity: " + last_extrinsics_sanity_)
                             : "degraded or dropped data detected";
        }

        auto add_kv = [&](const std::string& k, const std::string& v) {
            diagnostic_msgs::msg::KeyValue kv;
            kv.key = k;
            kv.value = v;
            st.values.push_back(std::move(kv));
        };

        const double proj_res_total_mean = (proj_res_samples_total_ > 0)
            ? (proj_res_sum_total_ / static_cast<double>(proj_res_samples_total_)) : 0.0;

        add_kv("cycles_completed", std::to_string(cycles_completed_));
        add_kv("camera_buffer_size", std::to_string(camera_profiles_[0].buffer.size()));
        add_kv("lidar_buffer_size", std::to_string(lidar_buffer_.size()));
        add_kv("motion_buffer_size", std::to_string(motion_buffer_.size()));
        add_kv("tracks_active", std::to_string(last_track_count_));
        add_kv("frame_history_size", std::to_string(frame_history_.size()));
        add_kv("last_observation_count", std::to_string(last_obs_count_));
        add_kv("last_latency_ms", fmt_double(last_latency_ms_, 2));
        add_kv("max_latency_ms", fmt_double(max_latency_ms_, 2));
        add_kv("last_ego_speed_mps", fmt_double(last_ego_speed_mps_, 3));
        add_kv("blind_distance_target_speed_mps", fmt_double(blind_distance_target_speed_mps_, 3));
        add_kv("last_blind_distance_current_m", fmt_double(last_blind_distance_current_m_, 3));
        add_kv("last_blind_distance_target_m", fmt_double(last_blind_distance_target_m_, 3));
        add_kv("max_blind_distance_target_m", fmt_double(max_blind_distance_target_m_, 3));
        add_kv("last_camera_lidar_delta_ms", fmt_double(last_camera_lidar_delta_ms_, 3));
        add_kv("last_camera_lidar_abs_delta_ms", fmt_double(last_camera_lidar_abs_delta_ms_, 3));
        add_kv("max_camera_lidar_abs_delta_ms", fmt_double(max_camera_lidar_abs_delta_ms_, 3));
        add_kv("camera_lidar_delta_samples", std::to_string(camera_lidar_delta_samples_));
        add_kv("sync_delta_mean_ms", fmt_double(sync_delta_mean_ms_, 3));
        add_kv("sync_delta_std_ms", fmt_double(sync_delta_std_ms_, 3));
        add_kv("sync_delta_abs_p95_ms", fmt_double(sync_delta_abs_p95_ms_, 3));
        add_kv("last_motion_age_ms", fmt_double(last_motion_age_ms_, 3));
        add_kv("max_motion_age_ms", fmt_double(max_motion_age_ms_, 3));
        add_kv("last_motion_comp_error_m", fmt_double(last_motion_comp_error_m_, 3));
        add_kv("max_motion_comp_error_m", fmt_double(max_motion_comp_error_m_, 3));
        add_kv("last_projection_residual_mean_px", fmt_double(last_projection_residual_mean_px_, 3));
        add_kv("projection_residual_max_px", fmt_double(proj_res_max_px_, 3));
        add_kv("projection_residual_samples_total", std::to_string(proj_res_samples_total_));
        add_kv("projection_residual_mean_total_px", fmt_double(proj_res_total_mean, 3));
        add_kv("use_tf_extrinsics", use_tf_extrinsics_ ? "true" : "false");
        add_kv("last_extrinsics_source", last_extrinsics_source_);
        add_kv("last_extrinsics_sanity", last_extrinsics_sanity_);
        add_kv("last_tf_lookup_ok", last_tf_lookup_ok_ ? "true" : "false");
        add_kv("last_tf_age_ms", fmt_double(last_tf_age_ms_, 3));
        add_kv("last_tf_lidar_frame", last_tf_lidar_frame_);
        add_kv("last_tf_camera_frame", last_tf_camera_frame_);
        add_kv("last_tf_lookup_error", last_tf_lookup_error_);
        add_kv("last_health", std::to_string(last_health_));
        add_kv("registered_cameras", std::to_string(camera_profiles_.size()));
        add_kv("registered_lidars", std::to_string(lidar_profiles_.size()));
        add_kv("enable_corridor_output", enable_corridor_output_ ? "true" : "false");
        add_kv("enable_debug_viz", enable_debug_viz_ ? "true" : "false");
        add_kv("enable_online_calibration", enable_online_calibration_ ? "true" : "false");
        add_kv("enable_triangular_cone_mask", primary_ext_.enable_triangular_cone_mask ? "true" : "false");
        add_kv("enable_stereo_depth_fusion", enable_stereo_depth_fusion_ ? "true" : "false");
        add_kv("replay_anchor_stamp", has_replay_anchor_ ? fmt_double(replay_anchor_stamp_, 6) : "none");

        for (const auto& kv : last_stage_ms_) add_kv("stage_ms_" + kv.first, fmt_double(kv.second, 3));
        for (const auto& kv : max_stage_ms_) add_kv("stage_ms_max_" + kv.first, fmt_double(kv.second, 3));

        add_kv("camera_overflow", std::to_string(cnt_camera_overflow_));
        add_kv("lidar_overflow", std::to_string(cnt_lidar_overflow_));
        add_kv("motion_overflow", std::to_string(cnt_motion_overflow_));
        add_kv("oosm_within_budget", std::to_string(cnt_oosm_within_budget_));
        add_kv("oosm_dropped", std::to_string(cnt_oosm_dropped_));
        add_kv("camera_miss", std::to_string(cnt_camera_miss_));
        add_kv("tracks_created", std::to_string(cnt_tracks_created_));
        add_kv("tracks_deleted", std::to_string(cnt_tracks_deleted_));
        add_kv("assoc_candidates", std::to_string(cnt_assoc_candidates_));
        add_kv("assoc_matches", std::to_string(cnt_assoc_matches_));
        add_kv("assoc_rejects", std::to_string(cnt_assoc_rejects_));
        add_kv("replay_requests", std::to_string(cnt_replay_requests_));
        add_kv("replay_completed", std::to_string(cnt_replay_completed_));
        add_kv("replay_frames", std::to_string(cnt_replay_frames_));
        add_kv("replay_budget_exceeded", std::to_string(cnt_replay_budget_exceeded_));
        add_kv("replay_no_frames", std::to_string(cnt_replay_no_frames_));
        add_kv("replay_no_base_snapshot", std::to_string(cnt_replay_no_base_snapshot_));
        add_kv("projection_projected_samples", std::to_string(cnt_proj_samples_));
        add_kv("projection_matched_samples", std::to_string(cnt_matched_samples_));
        add_kv("projection_clusters_no_match", std::to_string(cnt_clusters_no_match_));
        add_kv("projection_high_residual_clusters", std::to_string(cnt_high_res_clusters_));
        add_kv("frustum_recovered_observations", std::to_string(cnt_frustum_recovered_));
        add_kv("stereo_depth_frames", std::to_string(cnt_stereo_depth_frames_));
        add_kv("stereo_depth_fused_observations", std::to_string(cnt_stereo_fused_));
        add_kv("camera_fallback_frames", std::to_string(cnt_cam_fb_frames_));
        add_kv("camera_fallback_observations", std::to_string(cnt_cam_fb_obs_));
        add_kv("camera_fallback_no_camera", std::to_string(cnt_cam_fb_no_cam_));
        add_kv("camera_fallback_stale_camera", std::to_string(cnt_cam_fb_stale_));
        add_kv("camera_fallback_duplicate_frame", std::to_string(cnt_cam_fb_dup_));
        add_kv("tf_lookup_success", std::to_string(cnt_tf_lookup_success_));
        add_kv("tf_lookup_fail", std::to_string(cnt_tf_lookup_fail_));
        add_kv("tf_extrinsics_stale", std::to_string(cnt_tf_extrinsics_stale_));
        add_kv("tf_extrinsics_missing_frame", std::to_string(cnt_tf_extrinsics_missing_frame_));
        add_kv("motion_missing_for_covariance", std::to_string(cnt_motion_missing_for_cov_));
        add_kv("online_calib_runs", std::to_string(cnt_online_calib_runs_));
        add_kv("online_calib_applied", std::to_string(cnt_online_calib_applied_));
        add_kv("corridor_published", std::to_string(cnt_corridor_published_));
        add_kv("debug_markers_published", std::to_string(cnt_debug_markers_published_));
        add_kv("extra_lidar_messages", std::to_string(cnt_extra_lidar_messages_));
        add_kv("camera_info_updates", std::to_string(cnt_camera_info_updates_));

        msg.status.push_back(std::move(st));
        pub_diag_->publish(msg);
    }

    std::mutex mutex_;

    // Parameters
    std::string camera_topic_, lidar_topic_, motion_topic_;
    std::string motion_msg_type_{"odometry"};
    std::string obs_out_topic_, track_out_topic_, diag_topic_, target_frame_;
    bool enable_projection_semantics_{true};
    bool use_tf_extrinsics_{false};
    std::string tf_extrinsics_lidar_frame_, tf_extrinsics_camera_frame_;
    double tf_lookup_timeout_ms_{20.0}, tf_extrinsics_warn_age_ms_{40.0}, tf_extrinsics_critical_age_ms_{120.0};
    double projection_residual_warn_px_{5.0}, projection_residual_critical_px_{12.0};
    bool enable_hungarian_early_fusion_{true};
    bool enable_chassis_attitude_compensation_{true};
    double aero_pitch_rad_per_mps2_{0.0}, aero_tz_m_per_mps2_{0.0};
    bool enable_bidirectional_frustum_fusion_{true};
    double frustum_min_confidence_{0.55}, frustum_dedup_radius_m_{1.2};
    bool enable_ground_plane_depth_fusion_{true};
    double ground_z_base_m_{-0.22}, ground_slope_rad_{0.0};
    bool enable_stereo_depth_fusion_{true};
    std::string depth_topic_, secondary_depth_topic_;
    double stereo_depth_scale_{0.0}, stereo_baseline_m_{0.12}, stereo_disparity_sigma_px_{0.5};
    double stereo_depth_max_age_ms_{150.0}, stereo_depth_min_m_{0.5}, stereo_depth_max_m_{45.0};
    double max_rewind_s_{0.15};
    int32_t max_replay_steps_{8}, frame_history_size_{128};
    double max_camera_lidar_delta_s_{0.12};
    double sync_offset_warn_ms_{40.0}, sync_offset_critical_ms_{90.0};
    int32_t sync_stats_window_size_{60};
    double sync_jitter_warn_ms_{10.0}, sync_jitter_critical_ms_{25.0}, camera_retention_s_{2.0};
    bool enable_camera_fallback_{true};
    double camera_fallback_max_age_s_{0.25}, camera_fallback_min_confidence_{0.25};
    double camera_fallback_min_bbox_height_px_{8.0}, camera_fallback_cone_height_m_{0.30};
    double camera_fallback_min_depth_m_{2.0}, camera_fallback_max_depth_m_{35.0};
    double camera_fallback_quality_scale_{0.45}, camera_fallback_cov_xy_m2_{4.0}, camera_fallback_cov_z_m2_{8.0};
    int32_t queue_size_camera_{16}, queue_size_lidar_{16}, queue_size_motion_{64};
    std::string drop_policy_{"drop_oldest"};
    double blind_distance_target_speed_mps_{16.0};
    double motion_age_warn_ms_{80.0}, motion_age_critical_ms_{180.0}, motion_comp_error_gain_{0.6};
    double process_rate_hz_{30.0}, diag_rate_hz_{2.0};
    bool enable_tracks_{true}, enable_lidar_stripe_semantics_{true};
    bool require_camera_for_single_point_clusters_{true}, assume_uncounted_clusters_valid_{true};
    int32_t min_uncorroborated_cluster_points_{2};
    bool enable_corridor_output_{false};
    std::string corridor_markers_topic_, corridor_path_topic_;
    double corridor_horizon_m_{20.0}, corridor_step_m_{1.0};
    bool enable_debug_viz_{false};
    std::string debug_markers_topic_, debug_reprojection_topic_;
    bool enable_online_calibration_{false};
    int32_t online_calibration_window_frames_{30}, online_calibration_min_pairs_{12};
    bool online_calibration_apply_extrinsics_{false};
    std::string secondary_camera_topic_, camera_info_topic_, secondary_camera_info_topic_, camera_profiles_json_;
    bool enable_camera_info_intrinsics_{true};
    double secondary_camera_fx_{0.0}, secondary_camera_fy_{0.0}, secondary_camera_cx_{0.0}, secondary_camera_cy_{0.0};
    double secondary_tx_m_{0.0}, secondary_ty_m_{0.0}, secondary_tz_m_{0.0};
    double secondary_roll_rad_{0.0}, secondary_pitch_rad_{0.0}, secondary_yaw_rad_{0.0};
    std::vector<std::string> lidar_topics_extra_;
    std::string lidar_profiles_json_;
    double health_degrade_scale_{1.5}, quality_warn_threshold_{0.4}, quality_critical_threshold_{0.2};
    int32_t sensor_qos_depth_{10}, output_qos_depth_{20}, diag_qos_depth_{10};

    CameraIntrinsicsC primary_intr_{};
    ExtrinsicsC primary_ext_{};
    TrackerConfigC trk_cfg_{};
    void* tracker_handle_{nullptr};
    void* history_base_snapshot_{nullptr};

    std::vector<CameraProfileState> camera_profiles_;
    std::vector<LidarProfileState> lidar_profiles_;
    std::deque<std::pair<double, fusion_interfaces::msg::LidarConeClusters::ConstSharedPtr>> lidar_buffer_;
    std::deque<std::pair<double, MotionSampleC>> motion_buffer_;
    std::deque<FrameRecordC> frame_history_;
    std::deque<double> sync_delta_window_ms_;
    std::deque<OnlineCalibPairC> calib_pairs_window_;
    int32_t online_calib_frames_since_run_{0};

    // Runtime state
    double last_processed_lidar_stamp_{0.0}, last_tracker_stamp_{-1.0};
    double last_camera_fallback_stamp_{-1.0};
    bool pending_replay_{false}, has_replay_anchor_{false};
    double replay_anchor_stamp_{0.0};
    double last_latency_ms_{0.0}, max_latency_ms_{0.0};
    int32_t last_obs_count_{0}, last_track_count_{0}, last_health_{kSensorHealthOk};
    int64_t cycles_completed_{0};
    double last_ego_speed_mps_{0.0}, last_ego_pitch_rad_{0.0}, last_ego_roll_rad_{0.0};
    double last_blind_distance_current_m_{0.0}, last_blind_distance_target_m_{0.0}, max_blind_distance_target_m_{0.0};
    double last_camera_lidar_delta_ms_{0.0}, last_camera_lidar_abs_delta_ms_{0.0}, max_camera_lidar_abs_delta_ms_{0.0};
    int64_t camera_lidar_delta_samples_{0};
    double sync_delta_mean_ms_{0.0}, sync_delta_std_ms_{0.0}, sync_delta_abs_p95_ms_{0.0};
    double last_projection_residual_mean_px_{0.0}, proj_res_sum_total_{0.0}, proj_res_max_px_{0.0};
    int64_t proj_res_samples_total_{0};
    double last_motion_age_ms_{0.0}, max_motion_age_ms_{0.0}, last_motion_comp_error_m_{0.0}, max_motion_comp_error_m_{0.0};
    bool last_tf_lookup_ok_{true};
    double last_tf_age_ms_{0.0};
    std::string last_extrinsics_source_{"params"}, last_extrinsics_sanity_{"ok"}, last_tf_lookup_error_, last_tf_lidar_frame_, last_tf_camera_frame_;
    int32_t extrinsics_sanity_warn_streak_{0};
    std::map<std::string, double> last_stage_ms_, max_stage_ms_;

    // Counters
    int64_t cnt_camera_overflow_{0}, cnt_lidar_overflow_{0}, cnt_motion_overflow_{0};
    int64_t cnt_oosm_within_budget_{0}, cnt_oosm_dropped_{0}, cnt_camera_miss_{0};
    int64_t cnt_tracks_created_{0}, cnt_tracks_deleted_{0};
    int64_t cnt_assoc_candidates_{0}, cnt_assoc_matches_{0}, cnt_assoc_rejects_{0};
    int64_t cnt_replay_requests_{0}, cnt_replay_completed_{0}, cnt_replay_frames_{0};
    int64_t cnt_replay_budget_exceeded_{0}, cnt_replay_no_frames_{0}, cnt_replay_no_base_snapshot_{0};
    int64_t cnt_proj_samples_{0}, cnt_matched_samples_{0}, cnt_clusters_no_match_{0}, cnt_high_res_clusters_{0};
    int64_t cnt_frustum_recovered_{0}, cnt_stereo_depth_frames_{0}, cnt_stereo_fused_{0};
    int64_t cnt_cam_fb_frames_{0}, cnt_cam_fb_obs_{0}, cnt_cam_fb_no_cam_{0}, cnt_cam_fb_stale_{0}, cnt_cam_fb_dup_{0};
    int64_t cnt_tf_lookup_success_{0}, cnt_tf_lookup_fail_{0}, cnt_tf_extrinsics_stale_{0}, cnt_tf_extrinsics_missing_frame_{0};
    int64_t cnt_motion_missing_for_cov_{0}, cnt_online_calib_runs_{0}, cnt_online_calib_applied_{0};
    int64_t cnt_corridor_published_{0}, cnt_debug_markers_published_{0};
    int64_t cnt_extra_lidar_messages_{0}, cnt_camera_info_updates_{0};

    // ROS interfaces
    rclcpp::Publisher<fusion_interfaces::msg::ConeObservation>::SharedPtr pub_obs_;
    rclcpp::Publisher<fusion_interfaces::msg::ConeTrack>::SharedPtr pub_tracks_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_corridor_markers_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_corridor_path_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_debug_markers_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_debug_reprojection_;

    std::vector<rclcpp::Subscription<fusion_interfaces::msg::CameraConeDetections>::SharedPtr> sub_cams_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> sub_cam_infos_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> sub_depths_;
    rclcpp::Subscription<fusion_interfaces::msg::LidarConeClusters>::SharedPtr sub_lidar_;
    std::vector<rclcpp::Subscription<fusion_interfaces::msg::LidarConeClusters>::SharedPtr> sub_extra_lidars_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_motion_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_motion_twist_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_motion_twist_cov_;

#ifdef FUSION_HAS_TF2_ROS
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
#endif

    rclcpp::TimerBase::SharedPtr process_timer_;
    rclcpp::TimerBase::SharedPtr diag_timer_;
};

}  // namespace fusion_core_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(fusion_core_cpp::OusterLidarComposableNode)
RCLCPP_COMPONENTS_REGISTER_NODE(fusion_core_cpp::FusionCoreComposableNode)
