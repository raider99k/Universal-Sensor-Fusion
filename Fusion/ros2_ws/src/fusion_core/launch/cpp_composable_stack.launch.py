from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Zero-copy C++20 composable pipeline: OusterLidarComposableNode + FusionCoreComposableNode."""
    lidar_params = LaunchConfiguration("lidar_params_file")
    fusion_params = LaunchConfiguration("fusion_params_file")
    onnx_params = LaunchConfiguration("onnx_params_file")
    use_onnx_camera = LaunchConfiguration("use_onnx_camera")
    use_visualizer = LaunchConfiguration("use_visualizer")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "lidar_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_adapters"), "config", "lidar_adapter.params.yaml"]
                ),
                description="YAML parameters for OusterLidarComposableNode.",
            ),
            DeclareLaunchArgument(
                "fusion_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "fusion.params.yaml"]
                ),
                description="YAML parameters for FusionCoreComposableNode.",
            ),
            DeclareLaunchArgument(
                "onnx_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_adapters"), "config", "onnx_camera_detector.params.yaml"]
                ),
                description="YAML parameters for onnx_camera_detector (swap to trt_dla_camera_detector.params.yaml on Jetson).",
            ),
            DeclareLaunchArgument(
                "use_onnx_camera",
                default_value="true",
                description="Launch bundled ONNX/TensorRT camera cone detector alongside the C++20 container.",
            ),
            DeclareLaunchArgument(
                "use_visualizer",
                default_value="false",
                description="Launch cone_visualizer_node for RViz2/Foxglove 3D track markers.",
            ),
            ComposableNodeContainer(
                name="fusion_cpp_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                output="screen",
                composable_node_descriptions=[
                    ComposableNode(
                        package="fusion_core_cpp",
                        plugin="fusion_core_cpp::OusterLidarComposableNode",
                        name="ouster_lidar_cpp_adapter",
                        parameters=[lidar_params],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                    ComposableNode(
                        package="fusion_core_cpp",
                        plugin="fusion_core_cpp::FusionCoreComposableNode",
                        name="fusion_core_cpp_node",
                        parameters=[fusion_params],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                ],
            ),
            Node(
                condition=IfCondition(use_onnx_camera),
                package="fusion_adapters",
                executable="onnx_camera_detector",
                name="onnx_camera_detector",
                output="screen",
                parameters=[onnx_params],
            ),
            Node(
                condition=IfCondition(use_visualizer),
                package="fusion_core",
                executable="cone_visualizer_node",
                name="cone_visualizer_node",
                output="screen",
            ),
        ]
    )
