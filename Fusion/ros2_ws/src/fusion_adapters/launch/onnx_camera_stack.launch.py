from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Camera pipeline with the bundled ONNX detector in place of an external detector.

    image topic -> onnx_camera_detector -> /camera/cone_detections -> fusion_node.
    Swap it for any other detector by not launching this file and publishing
    Detection2DArray into camera_detection_adapter instead.
    """
    onnx_params = LaunchConfiguration("onnx_params_file")
    lidar_params = LaunchConfiguration("lidar_params_file")
    fusion_params = LaunchConfiguration("fusion_params_file")
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)

    def share(pkg, *parts):
        return PathJoinSubstitution([FindPackageShare(pkg), *parts])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use /clock simulation or rosbag replay time.",
            ),
            DeclareLaunchArgument(
                "onnx_params_file",
                default_value=share("fusion_adapters", "config", "onnx_camera_detector.params.yaml"),
            ),
            DeclareLaunchArgument(
                "lidar_params_file",
                default_value=share("fusion_adapters", "config", "lidar_adapter.params.yaml"),
            ),
            DeclareLaunchArgument(
                "fusion_params_file",
                default_value=share("fusion_core", "config", "fusion.params.yaml"),
            ),
            Node(
                package="fusion_adapters",
                executable="onnx_camera_detector",
                name="onnx_camera_detector",
                output="screen",
                parameters=[onnx_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="fusion_adapters",
                executable="ouster_lidar_adapter",
                name="ouster_lidar_adapter",
                output="screen",
                parameters=[lidar_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="fusion_core",
                executable="fusion_node",
                name="fusion_node",
                output="screen",
                parameters=[fusion_params, {"use_sim_time": use_sim_time}],
            ),
        ]
    )
