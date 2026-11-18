# Universal Sensor Fusion: ROS 2 Perception Stack for Formula Student Driverless

A modular ROS 2 perception and multi-target sensor fusion stack developed for Formula Student Driverless (FSD) and autonomous racing. Combines N-camera/M-LiDAR spatio-temporal early fusion with an $SE(2)$ static landmark Extended Kalman Filter (EKF), clothoid corridor estimation, trackside auto-calibration, and a Dirichlet belief distribution.

Runs entirely on CPU (Linux, WSL2, Docker) without requiring discrete GPU or CUDA runtime dependencies, with native C++20 acceleration (`fusion_core_cpp`) for sub-6ms OS1-64 point cloud processing, ~0.11ms batch early fusion, ~0.09ms EKF tracking, and zero-copy `rclcpp_components` execution.

---

## Architecture Overview

```
+--------------------------------------------------------------------------------------------------+
|                                    Synthetic Track Simulation                                    |
|  [Track Generator] -> (3D cone positions in odom, closed loop vehicle motion)                    |
|    |--> /vision/cone_detections (vision_msgs/msg/Detection2DArray)                               |
|    |--> /ouster/points          (sensor_msgs/msg/PointCloud2)                                    |
|    |--> /ego/odometry           (nav_msgs/msg/Odometry @ 100 Hz)                                 |
|    +--> /ground_truth/cones     (visualization_msgs/msg/MarkerArray)                             |
+--------------------------------------------------------------------------------------------------+
                                          |                                  |
                                          v                                  v
+--------------------------------------------------------------------------------------------------+
|                                  Perception Adapters (fusion_adapters)                           |
|  Camera:                                                                                         |
|    - onnx_camera_detector / trt_camera_detector: ONNX Runtime (YOLO26n nano.onnx, CPU/CUDA/TRT)  |
|    - camera_detection_adapter: bridges vision_msgs/Detection2DArray -> /camera/cone_detections   |
|  LiDAR:                                                                                          |
|    - ouster_lidar_adapter / OusterLidarComposableNode (C++20): OS1/VLP-16 -> /lidar/cone_clusters|
|    - lidar_cluster_adapter: bridges PoseArray, MarkerArray, Detection3DArray                     |
+--------------------------------------------------------------------------------------------------+
                                          |                                  |
                                          +-----------------+----------------+
                                                            |
                                                            v
+--------------------------------------------------------------------------------------------------+
|                        Fusion Core (fusion_node / FusionCoreComposableNode)                      |
|  - N-camera & M-LiDAR registry, auto CameraInfo intrinsics, tf2 extrinsics & online calibration  |
|  - Ring buffers (drop_oldest) & bounded OOSM rewind/replay (<= 150 ms horizon)                   |
|  - Early Fusion: C++20/NumPy 7-point & frustum projection, Brown-Conrady distortion, attitude    |
|  - EKF Tracker: SE(2) static landmark kinematics, Hungarian assignment, Dirichlet & corridor     |
|  - Single-sensor fallback: ground-plane homography ray projection & LiDAR stripe classification  |
|  - Published Topics (>= 20 Hz):                                                                  |
|      /perception/cone_observations  (fusion_interfaces/msg/ConeObservation)                      |
|      /perception/cone_tracks        (fusion_interfaces/msg/ConeTrack)                            |
|      /perception/diagnostics        (diagnostic_msgs/msg/DiagnosticArray)                        |
|      /perception/corridor_{markers,path} & /perception/debug_{markers,reprojection} (optional)   |
+--------------------------------------------------------------------------------------------------+
                                                            |
                                                            v
+--------------------------------------------------------------------------------------------------+
|                                       Visualization & Telemetry                                  |
|  - cone_visualizer_node: converts tracks & observations to RViz2 MarkerArray                      |
|  - Foxglove Studio Layout (config/foxglove_studio_layout.json via ws://localhost:8765)           |
|  - RViz2 Configuration    (config/fusion_view.rviz)                                              |
+--------------------------------------------------------------------------------------------------+
```

---

## Benchmarks

Evaluated on real Formula Student Driverless rosbags (Mainfranken Racing 2026, AMZ Driverless 2017) against surveyed RTK-GPS ground truth and on-car/constant-velocity baselines, plus a deterministic synthetic autocross stream. All timings are CPU-only (no GPU/CUDA).

### Real Vehicle Datasets (Mainfranken Racing 2026 — Track 1 vs. Surveyed RTK-GPS)

Sensors: Ouster OS1-64 (64-beam LiDAR), ZED2 stereo camera, SBG IMU/velocity.

| Pipeline Configuration | Position RMSE | Overall Recall | Recall (12–21 m) | False Positive Rate | Semantic Accuracy |
|---|---|---|---|---|---|
| On-car stereo baseline (standalone) | 10.48 cm (`0.1048 m`) | 96.92% | 96.40% | 57.11% | 50.87% |
| Baseline early fusion | — | — | — | — | 88.02% |
| **This stack (`fusion_core` + `nano.onnx` YOLO26n + OS1-64)** | **4.85 cm** (`0.0485 m`) | **97.69%** | **96.68%** | **5.47%** | **86.84%** |
| **This stack (`fusion_core` + OS1-64, 66 scans)** | **4.91 cm** (`0.0491 m`) | **99.23%** | **98.61%** | **0.72%** | **98.54%** |
| **Multi-camera `FusionNode` + online cal (`66` OS1-64 + ZED2)** | 14.91 cm (`0.1491 m`) | **97.84%** | **96.95%** | **1.84%** | **87.93%** |

*(195 ZED2 frames evaluated for the end-to-end `nano.onnx` run; 66 64-beam scans evaluated for OS1-64 + RTK-GPS ground truth; online trackside calibration reduced reprojection RMSE from `70.32 px` to `27.71 px` across `239` matched pairs and estimated `dt_sync = -0.08 s`.)*

### Tracking, Corridor Estimation & Cornering Stability (vs. Constant-Velocity Baseline)

| Dataset | Metric | Constant-Velocity Baseline | This Stack ($SE(2)$ EKF + C++20) |
|---|---|---|---|
| **Mainfranken 2026 FSCZ** (10 laps, 500 frames) | Track innovation RMSE | `1.3853 m` | **`0.1391 m`** |
| Mainfranken 2026 FSCZ (10 laps, 500 frames) | High-G cornering RMSE | `1.4812 m` | **`0.1401 m`** |
| Mainfranken 2026 FSCZ (10 laps, 500 frames) | Semantic flips (1,261 landmarks) | `117` | **`10`** |
| Mainfranken 2026 FSCZ (10 laps, 500 frames) | Mean corridor confidence / C++ EKF pos diff | — | **`0.3091`** / **`0.0 m`** |
| **AMZ 2017 Zurich Airfield** (VLP-16, 50 active frames) | Track innovation RMSE | `1.1292 m` | **`0.0243 m`** (`-97.85%`) |
| AMZ 2017 Zurich Airfield (VLP-16, 50 active frames) | High-G cornering RMSE | `1.0878 m` | **`0.0347 m`** (`-96.81%`) |
| AMZ 2017 Zurich Airfield (VLP-16, 50 active frames) | Long-range cone hits (12–28 m) | `0` | **`86`** (`+86`) |

### Synthetic Autocross Stream (80 Frames, Hairpin + Chicane, Seed 42)

| Metric | Legacy Baseline | This Stack ($SE(2)$ Fusion) |
|---|---|---|
| Position RMSE | `0.3509 m` | **`0.1188 m`** |
| Overall Recall | `48.99%` | **`96.84%`** |
| Long-Range Recall (15–35 m) | `0.00%` | **`93.06%`** |
| Semantic Accuracy | `79.97%` | **`99.58%`** |
| Track ID Switches | `313` | **`0`** |

### CPU Step Latency & C++20 Kernel Parity

| Workload | Prior / Unvectorized | Python / NumPy / ORT | C++20 `fusion_fast_kernels` |
|---|---|---|---|
| **Ouster OS1-64** (131,072 pts/scan + EKF) | `21.79 ms` | `8.76 ms` (NumPy) | **`5.70 ms`** (**3.82x faster**) |
| **Velodyne VLP-16** (16-beam + EKF) | `15.72 ms` | **`6.24 ms`** (NumPy) | — |
| **Batch Early Fusion** (Track 1, 66 scans) | — | `scipy` / NumPy | **`0.1065 ms`** (`0` mismatches) |
| **$SE(2)$ EKF Tracker Step** (Track 1, 66 scans) | — | `4.1881 ms` (Python) | **`0.0866 ms`** (**48.4x**, `0.0 m` diff) |
| **$SE(2)$ EKF Tracker Step** (FSCZ, 500 frames) | — | `2.0289 ms` (Python) | **`0.0990 ms`** (**20.5x**, `0.0 m` diff) |
| **Synthetic Autocross** (80 frames) | — | `8.91 ms` mean, `19.42 ms` P99 (NumPy) | — |
| **Bundled `nano.onnx` (YOLO26n)** camera inference | — | `49.99 ms` mean (ONNX Runtime CPU) | — |

### Live ROS 2 Rolling Runtime & Automated Test Verification (WSL2 Ubuntu 24.04)

| Verification Gate | Environment | Observed Result |
|---|---|---|
| **ROS 2 `fusion_regression_suite`** (`use_adapters:=true`, `rmw_zenoh_cpp`) | WSL2 ROS 2 Rolling (`vision_msgs` + `ouster_lidar_adapter`) | **PASS** — Conformance: `camera=299` (`19.67 Hz`), `lidar=293` (`19.47 Hz`), `motion=1460` (`97.00 Hz`), `0` invalid; OOSM: `diag=360`, `p99=159.61 ms`, `max=174.12 ms`, `0` drops, `102` replay requests / `67` completed |
| **ROS 2 `fusion_regression_suite`** (`use_adapters:=false`, `rmw_zenoh_cpp`) | WSL2 ROS 2 Rolling | **PASS** — Conformance: `camera=297` (`19.67 Hz`), `lidar=299` (`19.87 Hz`), `motion=1505` (`100.00 Hz`), `0` invalid; OOSM: `diag=369`, `p99=169.43 ms`, `max=188.55 ms`, `0` drops, `97` replay requests / `59` completed |
| **C++20 Zero-Copy Composable Container** (`cpp_composable_stack.launch.py`) | WSL2 ROS 2 Rolling (`component_container_mt`) | **PASS** — `OusterLidarComposableNode` + `FusionCoreComposableNode` loaded cleanly; `74` C++ fusion cycles (`obs=456`, `trk=1146`, `mkr=1494`, `diag=16`) |
| **ONNX Camera Stack** (`onnx_camera_stack.launch.py`) | WSL2 ROS 2 Rolling (`onnxruntime 1.32.0`) | **PASS** — Launched `onnx_camera_detector`, `ouster_lidar_adapter`, and `fusion_node` cleanly (`0` errors; 4 `test_onnx_camera_detector.py` CPU ORT tests passed) |
| **Unit & Integration Test Suite (`pytest`)** | WSL2 ROS 2 Rolling / Windows host | **75 passed, 0 skipped** in WSL2 (incl. `rosbag2_py` `.mcap` roundtrip & `onnxruntime`) · **74 passed, 1 skipped** on Windows (missing `rosbag2_py`) |

---

## Quickstart

### Docker Environment

Launch the stack with synthetic track simulation, EKF tracking, and Foxglove WebSocket bridge:

```bash
git clone https://github.com/raider99k/Universal-Sensor-Fusion.git
cd Universal-Sensor-Fusion
docker compose up
```

The stack immediately publishes perception topics and serves telemetry over WebSocket port `8765`.

---

## Telemetry & Visual Inspection

### 1. Foxglove Studio
1. Open [Foxglove Studio](https://foxglove.dev/).
2. Select **Open Connection** -> **Foxglove WebSocket** -> Address: `ws://localhost:8765`.
3. Load layout: **Layout** -> **Import from file** -> Select `config/foxglove_studio_layout.json`.
4. Inspect 3D cones, vehicle trajectory, covariance ellipses, track IDs, and latency diagnostics.

### 2. RViz2 (Native ROS 2)
```bash
rviz2 -d config/fusion_view.rviz
```

---

## Output Topics

| Topic | Type | Rate | Description |
|---|---|---|---|
| `/perception/cone_tracks` | `fusion_interfaces/msg/ConeTrack` | $\ge 20$ Hz | Multi-target EKF fused tracks (position, velocity, covariance, class probabilities) |
| `/perception/cone_observations` | `fusion_interfaces/msg/ConeObservation` | $\ge 20$ Hz | Associated camera+LiDAR fused cone observations |
| `/perception/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | $\ge 20$ Hz | P99 latency, blind distance, sync offsets, stage timings, queue health |
| `/perception/cone_track_markers` | `visualization_msgs/msg/MarkerArray` | $\ge 20$ Hz | 3D cone markers with classification colors, track IDs, covariance ellipsoids |
| `/perception/cone_observation_markers` | `visualization_msgs/msg/MarkerArray` | $\ge 20$ Hz | 3D observation spheres |
| `/perception/corridor_markers` | `visualization_msgs/msg/MarkerArray` | On step (`enable_corridor_output`) | Estimated left/right track boundaries and clothoid centerline |
| `/perception/corridor_path` | `nav_msgs/msg/Path` | On step (`enable_corridor_output`) | Sampled centerline waypoints with heading & curvature |
| `/perception/debug_markers` | `visualization_msgs/msg/MarkerArray` | On step (`enable_debug_viz`) | 3D diagnostic & uncertainty markers |
| `/perception/debug_reprojection` | `std_msgs/msg/String` | On step (`enable_debug_viz`) | JSON 2D camera reprojection & frustum annotations |
| `/ground_truth/cones` | `visualization_msgs/msg/MarkerArray` | 2 Hz | Ground-truth track cones in `odom` frame |

> **Runtime Notes:**
> - **Docker (`ros:humble-ros-base`):** `docker compose up` includes `ros-humble-vision-msgs` and `ros-humble-foxglove-bridge` for `synthetic_stack` / `full_stack`. Install `onnxruntime` and `opencv-python` if launching `onnx_camera_stack.launch.py` inside that container.
> - **Jetson Orin AGX (TRT/NVDLA):** [trt_dla_camera_detector.params.yaml](Fusion/ros2_ws/src/fusion_adapters/config/trt_dla_camera_detector.params.yaml) enables `TensorrtExecutionProvider` (FP16 + DLA0) with automatic CUDA/CPU fallback; live TensorRT/NVDLA hardware execution remains pending target Jetson testing.
> - **WSL2 ROS 2 Rolling:** Default FastDDS SPDP discovery can miss `/ego/odometry` across WSL2 loopback interfaces; running `ros2 run rmw_zenoh_cpp rmw_zenohd` with `RMW_IMPLEMENTATION=rmw_zenoh_cpp` passed all live regression gates cleanly.

---

## Native Build & Testing

### 1. Build Workspace
```bash
source /opt/ros/humble/setup.bash
cd Fusion/ros2_ws
colcon build --packages-select fusion_interfaces fusion_core_cpp fusion_core fusion_adapters
source install/setup.bash
```

### 2. Verify Pipeline Launches
```bash
# Synthetic test track simulation (zero hardware needed)
ros2 launch fusion_core synthetic_stack.launch.py

# Zero-copy C++20 composable pipeline (OusterLidarComposableNode + FusionCoreComposableNode)
ros2 launch fusion_core cpp_composable_stack.launch.py

# Core fusion node standalone
ros2 launch fusion_core fusion.launch.py

# Full stack with camera and LiDAR adapters
ros2 launch fusion_adapters full_stack.launch.py

# Full stack with pre-bundled ONNX camera detector
ros2 launch fusion_adapters onnx_camera_stack.launch.py

# Offline trackside calibration (ground plane + 6-DoF extrinsics + clock sync)
ros2 run fusion_core fusion_auto_calibrate --help
```
