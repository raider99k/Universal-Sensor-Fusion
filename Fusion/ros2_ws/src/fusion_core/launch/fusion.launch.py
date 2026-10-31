from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "fusion.params.yaml"]
                ),
                description="Path to the fusion parameters YAML file.",
            ),
            Node(
                package="fusion_core",
                executable="fusion_node",
                name="fusion_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
