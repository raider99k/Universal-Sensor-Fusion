# ROS 2 Humble Fusion Workspace

This workspace contains:

- `fusion_interfaces`: message contracts for camera/LiDAR inputs and fused outputs.
- `fusion_core`: timestamp-driven fusion node with bounded queues, bounded replay, diagnostics, and EKF-based multi-target tracking with Mahalanobis + semantic gating.
- `fusion_adapters`: adapter nodes to connect typical ZED/Ouster ROS streams to fusion contracts.

## Verification & Maturity Status

- `fusion_interfaces`: Stable (message definitions verified across ROS 2 nodes and `.mcap` rosbag streams).
- `fusion_core` tracking: Verified ($SE(2)$ static landmark kinematics, Hungarian data association, Dirichlet epistemic belief filtering, Joseph-form covariance updates; verified against AMZ 2017 and Mainfranken 2026 surveyed RTK-GPS datasets).
- `fusion_core` early fusion: Verified (Brown-Conrady lens distortion and iterative undistortion, 7-point sigma sampling, dynamic pitch/roll attitude compensation, and 1-to-1 bounding box matching).
- `fusion_adapters`: Verified (zero-copy vectorized NumPy `PointCloud2` unpacking, continuous motion deskewing, concentric-zone ground modeling with outer-ring slope extrapolation, and $O(K)$ spatial clustering with neighbor bucket pruning).

## Build

```bash
cd Fusion/ros2_ws
colcon build --packages-select fusion_interfaces fusion_core fusion_adapters
source install/setup.bash
```

## Run

```bash
ros2 launch fusion_core fusion.launch.py
```

Run adapters + fusion together:

```bash
ros2 launch fusion_adapters full_stack.launch.py
```

Run synthetic producers + fusion (no external team code required):

```bash
ros2 launch fusion_core synthetic_stack.launch.py
```

## Plug-and-Play Integration

1. Keep your existing camera and LiDAR nodes running.
2. Ensure they publish messages compatible with:
   - `vision_msgs/msg/Detection2DArray` for camera adapter input
   - `sensor_msgs/msg/PointCloud2` for lidar adapter input
3. Tune adapter params if needed:
   - `fusion_adapters/config/camera_adapter.params.yaml`
   - `fusion_adapters/config/lidar_adapter.params.yaml`
4. Remap topics in `fusion_core/config/fusion.params.yaml` if needed.
5. Consume fused outputs:
   - `/perception/cone_observations`
   - `/perception/cone_tracks`
   - `/perception/diagnostics`
