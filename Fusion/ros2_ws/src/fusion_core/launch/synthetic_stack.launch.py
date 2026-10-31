from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # Configurations
    fusion_params = LaunchConfiguration("fusion_params_file")
    track_params = LaunchConfiguration("track_params_file")
    camera_adapter_params = LaunchConfiguration("camera_adapter_params_file")
    lidar_adapter_params = LaunchConfiguration("lidar_adapter_params_file")
    camera_params = LaunchConfiguration("camera_params_file")
    lidar_params = LaunchConfiguration("lidar_params_file")
    motion_params = LaunchConfiguration("motion_params_file")

    use_adapters = LaunchConfiguration("use_adapters")
    use_foxglove = LaunchConfiguration("use_foxglove_bridge")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_adapters",
                default_value="true",
                description="Run standard synthetic track with camera & lidar adapters.",
            ),
            DeclareLaunchArgument(
                "use_foxglove_bridge",
                default_value="true",
                description="Launch Foxglove WebSocket bridge on port 8765.",
            ),
            DeclareLaunchArgument(
                "fusion_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "fusion.params.yaml"]
                ),
                description="Fusion node params.",
            ),
            DeclareLaunchArgument(
                "track_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "synthetic_track.params.yaml"]
                ),
                description="Synthetic track and standard producer params.",
            ),
            DeclareLaunchArgument(
                "camera_adapter_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_adapters"), "config", "camera_adapter.params.yaml"]
                ),
                description="Camera detection adapter params.",
            ),
            DeclareLaunchArgument(
                "lidar_adapter_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_adapters"), "config", "lidar_adapter.params.yaml"]
                ),
                description="Lidar cluster adapter params.",
            ),
            DeclareLaunchArgument(
                "camera_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "synthetic_camera.params.yaml"]
                ),
                description="Legacy synthetic camera params.",
            ),
            DeclareLaunchArgument(
                "lidar_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "synthetic_lidar.params.yaml"]
                ),
                description="Legacy synthetic lidar params.",
            ),
            DeclareLaunchArgument(
                "motion_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fusion_core"), "config", "synthetic_motion.params.yaml"]
                ),
                description="Legacy synthetic motion params.",
            ),

            # --- Primary Track Harness Pipeline (use_adapters = true) ---
            # 1. Deterministic Synthetic Track Publisher (standard producer topics)
            Node(
                condition=IfCondition(use_adapters),
                package="fusion_core",
                executable="synthetic_track_publisher",
                name="synthetic_track_publisher",
                output="screen",
                parameters=[track_params],
            ),
            # 2. Camera Detection Adapter (vision_msgs -> CameraConeDetections)
            Node(
                condition=IfCondition(use_adapters),
                package="fusion_adapters",
                executable="camera_detection_adapter",
                name="camera_detection_adapter",
                output="screen",
                parameters=[camera_adapter_params],
            ),
            # 3. Ouster LiDAR Adapter (sensor_msgs/PointCloud2 -> LidarConeClusters)
            Node(
                condition=IfCondition(use_adapters),
                package="fusion_adapters",
                executable="ouster_lidar_adapter",
                name="ouster_lidar_adapter",
                output="screen",
                parameters=[lidar_adapter_params],
            ),

            # --- Legacy Direct Pipeline (use_adapters = false) ---
            Node(
                condition=UnlessCondition(use_adapters),
                package="fusion_core",
                executable="synthetic_camera_publisher",
                name="synthetic_camera_publisher",
                output="screen",
                parameters=[camera_params],
            ),
            Node(
                condition=UnlessCondition(use_adapters),
                package="fusion_core",
                executable="synthetic_lidar_publisher",
                name="synthetic_lidar_publisher",
                output="screen",
                parameters=[lidar_params],
            ),
            Node(
                condition=UnlessCondition(use_adapters),
                package="fusion_core",
                executable="synthetic_motion_publisher",
                name="synthetic_motion_publisher",
                output="screen",
                parameters=[motion_params],
            ),

            # --- Common Core Fusion Node ---
            Node(
                package="fusion_core",
                executable="fusion_node",
                name="fusion_node",
                output="screen",
                parameters=[fusion_params],
            ),

            # --- 3D Visualizer Marker Bridge ---
            Node(
                package="fusion_core",
                executable="cone_visualizer_node",
                name="cone_visualizer_node",
                output="screen",
            ),

            # --- Optional Foxglove WebSocket Bridge ---
            Node(
                condition=IfCondition(use_foxglove),
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[{
                    "port": 8765,
                    "address": "0.0.0.0",
                    "send_buffer_limit": 10000000,
                }],
            ),
        ]
    )
