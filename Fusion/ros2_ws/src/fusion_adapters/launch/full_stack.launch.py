from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    camera_params_file = LaunchConfiguration("camera_params_file")
    lidar_params_file = LaunchConfiguration("lidar_params_file")
    fusion_params_file = LaunchConfiguration("fusion_params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("fusion_adapters"),
                        "config",
                        "camera_adapter.params.yaml",
                    ]
                ),
                description="Path to camera adapter parameter YAML.",
            ),
            DeclareLaunchArgument(
                "lidar_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("fusion_adapters"),
                        "config",
                        "lidar_adapter.params.yaml",
                    ]
                ),
                description="Path to lidar adapter parameter YAML.",
            ),
            DeclareLaunchArgument(
                "fusion_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "fusion.params.yaml"]
                ),
                description="Path to fusion core parameter YAML.",
            ),
            Node(
                package="fusion_adapters",
                executable="camera_detection_adapter",
                name="camera_detection_adapter",
                output="screen",
                parameters=[camera_params_file],
            ),
            Node(
                package="fusion_adapters",
                executable="ouster_lidar_adapter",
                name="ouster_lidar_adapter",
                output="screen",
                parameters=[lidar_params_file],
            ),
            Node(
                package="fusion_core",
                executable="fusion_node",
                name="fusion_node",
                output="screen",
                parameters=[fusion_params_file],
            ),
        ]
    )
