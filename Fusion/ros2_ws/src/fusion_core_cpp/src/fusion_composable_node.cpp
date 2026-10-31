#include "fusion_core_cpp/fusion_fast_kernels.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "fusion_interfaces/msg/lidar_cone_cluster.hpp"
#include "fusion_interfaces/msg/lidar_cone_clusters.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace fusion_core_cpp {

class OusterLidarComposableNode : public rclcpp::Node {
public:
    explicit OusterLidarComposableNode(const rclcpp::NodeOptions& options)
        : rclcpp::Node("ouster_lidar_cpp_adapter", options) {
        pipe_cfg_.sensor_yaw_rad = this->declare_parameter("sensor_yaw_rad", 0.0);
        pipe_cfg_.sensor_offset_x_m = this->declare_parameter("sensor_offset_x_m", 0.0);
        pipe_cfg_.sensor_offset_y_m = this->declare_parameter("sensor_offset_y_m", 0.0);
        pipe_cfg_.sensor_offset_z_m = this->declare_parameter("sensor_offset_z_m", 0.0);
        pipe_cfg_.enable_imu_leveling = this->declare_parameter("enable_imu_leveling", true) ? 1 : 0;
        pipe_cfg_.roi_x_min = this->declare_parameter("roi_x_min", 1.2);
        pipe_cfg_.roi_x_max = this->declare_parameter("roi_x_max", 28.0);
        pipe_cfg_.roi_y_min = this->declare_parameter("roi_y_min", -10.0);
        pipe_cfg_.roi_y_max = this->declare_parameter("roi_y_max", 10.0);
        pipe_cfg_.roi_z_min = this->declare_parameter("roi_z_min", -0.60);
        pipe_cfg_.roi_z_max = this->declare_parameter("roi_z_max", 0.35);
        pipe_cfg_.enable_self_return_suppression = 1;
        pipe_cfg_.self_return_radius_m = 1.65;
        pipe_cfg_.self_return_z_min_m = -0.65;
        pipe_cfg_.self_return_z_max_m = 0.45;
        pipe_cfg_.self_return_min_range_m = 1.65;
        pipe_cfg_.max_points_for_clustering = 3500;
        pipe_cfg_.deskew_scan_duration_s = 0.05;
        pipe_cfg_.deskew_max_dt_s = 0.06;
        pipe_cfg_.deskew_spin_direction = 1.0;
        pipe_cfg_.enable_adaptive_ground_model = 1;
        pipe_cfg_.ground_z_max = this->declare_parameter("ground_z_max", -0.42);
        pipe_cfg_.ground_zone_radial_step_m = 2.5;
        pipe_cfg_.ground_zone_azimuth_bins = 24;
        pipe_cfg_.max_ground_slope_rad = 0.14;
        pipe_cfg_.ground_clearance_m = 0.065;
        pipe_cfg_.cone_max_radius_m = 0.25;
        pipe_cfg_.enable_ground_point_recovery = 1;
        pipe_cfg_.adaptive_range_ref_m = 12.0;
        pipe_cfg_.cluster_tolerance_m = 0.38;

        cls_cfg_.cluster_tolerance_m = pipe_cfg_.cluster_tolerance_m;
        cls_cfg_.min_cluster_points = 4;
        cls_cfg_.max_cluster_points = 260;
        cls_cfg_.enable_adaptive_clustering = 1;
        cls_cfg_.min_cluster_points_far = 2;
        cls_cfg_.adaptive_range_ref_m = 12.0;
        cls_cfg_.cone_min_height_m = 0.14;
        cls_cfg_.cone_max_height_m = 0.58;
        cls_cfg_.cone_max_radius_m = 0.25;
        cls_cfg_.covariance_floor_xy = 0.015;
        cls_cfg_.covariance_floor_z = 0.025;
        cls_cfg_.enable_surface_axis_offset = 1;
        cls_cfg_.surface_axis_offset_m = 0.045;
        cls_cfg_.enable_stripe_classification = 1;

        pub_clusters_ = this->create_publisher<fusion_interfaces::msg::LidarConeClusters>(
            "/perception/lidar/clusters", rclcpp::QoS(10));

        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->on_cloud(msg); });
    }

private:
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

        buf_x_.resize(4096);
        buf_y_.resize(4096);
        buf_z_.resize(4096);
        buf_refl_.resize(4096);
        LidarExtractStatsC stats{};
        const int32_t n_kept = cpp_extract_and_filter_ground(
            msg->data.data(), &pipe_cfg_,
            buf_x_.data(), buf_y_.data(), buf_z_.data(), buf_refl_.data(),
            4096, &stats);

        cls_cfg_.has_refl = (pipe_cfg_.off_stripe >= 0) ? 1 : 0;
        std::vector<ClusterDescriptorC> desc(128);
        const int32_t n_cls = cpp_cluster_and_build(
            n_kept, buf_x_.data(), buf_y_.data(), buf_z_.data(),
            cls_cfg_.has_refl ? buf_refl_.data() : nullptr,
            &cls_cfg_, desc.data(), 128);

        auto out_msg = std::make_unique<fusion_interfaces::msg::LidarConeClusters>();
        out_msg->header = msg->header;
        out_msg->clusters.reserve(n_cls);
        for (int32_t i = 0; i < n_cls; ++i) {
            fusion_interfaces::msg::LidarConeCluster c;
            c.centroid.x = desc[i].cx;
            c.centroid.y = desc[i].cy;
            c.centroid.z = desc[i].cz;
            c.covariance[0] = desc[i].var_x;
            c.covariance[1] = desc[i].cov_xy;
            c.covariance[3] = desc[i].cov_xy;
            c.covariance[4] = desc[i].var_y;
            c.covariance[8] = desc[i].var_z;
            c.geometry_quality = static_cast<float>(desc[i].geom_quality);
            c.point_count = static_cast<uint32_t>(desc[i].point_count);
            out_msg->clusters.push_back(c);
        }
        pub_clusters_->publish(std::move(out_msg));
    }

    LidarPipelineConfigC pipe_cfg_{};
    ClusterBuildConfigC cls_cfg_{};
    std::vector<double> buf_x_;
    std::vector<double> buf_y_;
    std::vector<double> buf_z_;
    std::vector<double> buf_refl_;
    rclcpp::Publisher<fusion_interfaces::msg::LidarConeClusters>::SharedPtr pub_clusters_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
};

}  // namespace fusion_core_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(fusion_core_cpp::OusterLidarComposableNode)
