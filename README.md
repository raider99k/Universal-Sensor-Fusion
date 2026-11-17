# Universal Sensor Fusion: ROS 2 Perception Stack for Formula Student Driverless

A modular ROS 2 perception and multi-target sensor fusion stack developed for Formula Student Driverless (FSD) and autonomous racing. Combines camera-LiDAR spatio-temporal early fusion with an $SE(2)$ static landmark Extended Kalman Filter (EKF) and Dirichlet belief distribution. 

Runs entirely on CPU (Linux, WSL2, Docker) without requiring discrete GPU or CUDA runtime dependencies, with optional C++20 acceleration for sub-6ms point cloud processing.

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
|    - onnx_camera_detector: ONNX Runtime (YOLO26n end2end) -> /camera/cone_detections              |
|    - camera_detection_adapter: bridges vision_msgs/Detection2DArray -> /camera/cone_detections   |
|  LiDAR:                                                                                          |
|    - ouster_lidar_adapter: OS1/VLP-16 deskewing + ground removal -> /lidar/cone_clusters         |
|    - lidar_cluster_adapter: bridges PoseArray, MarkerArray, Detection3DArray                     |
+--------------------------------------------------------------------------------------------------+
                                          |                                  |
                                          +-----------------+----------------+
                                                            |
                                                            v
+--------------------------------------------------------------------------------------------------+
|                                      Fusion Core (fusion_node)                                   |
|  - Ring buffers (drop_oldest) & bounded OOSM rewind/replay (<= 150 ms horizon)                   |
|  - Early Fusion: 7-point sigma projection, Brown-Conrady distortion, attitude compensation       |
|  - EKF Tracker: SE(2) static landmark kinematics, Hungarian assignment, Dirichlet belief update |
|  - Single-sensor fallback: ground-plane homography ray projection & LiDAR stripe classification  |
|  - Published Topics (>= 20 Hz):                                                                  |
|      /perception/cone_observations (fusion_interfaces/msg/ConeObservation)                       |
|      /perception/cone_tracks       (fusion_interfaces/msg/ConeTrack)                             |
|      /perception/diagnostics       (diagnostic_msgs/msg/DiagnosticArray)                         |
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

*(195 ZED2 frames evaluated for the end-to-end `nano.onnx` run; 66 64-beam scans evaluated for the OS1-64 + RTK-GPS ground-truth run.)*

### Tracking & Cornering Stability (vs. Constant-Velocity EKF Baseline)

| Dataset | Metric | Constant-Velocity Baseline | This Stack ($SE(2)$ EKF) |
|---|---|---|---|
| **Mainfranken 2026 FSCZ** (10 laps, 500 frames) | Track innovation RMSE | `1.3853 m` | **`0.1391 m`** |
| Mainfranken 2026 FSCZ (10 laps, 500 frames) | High-G cornering RMSE | `1.4812 m` | **`0.1401 m`** |
| Mainfranken 2026 FSCZ (10 laps, 500 frames) | Semantic flips (1,261 landmarks) | `117` | **`10`** |
| **AMZ 2017 Zurich Airfield** (VLP-16, 50 active frames) | Track innovation RMSE | `1.1292 m` | **`0.0243 m`** |
| AMZ 2017 Zurich Airfield (VLP-16, 50 active frames) | High-G cornering RMSE | `1.0878 m` | **`0.0347 m`** |
| AMZ 2017 Zurich Airfield (VLP-16, 50 active frames) | Long-range cone hits (12–28 m) | `0` | **`86`** |

### Synthetic Autocross Stream (80 Frames, Hairpin + Chicane, Seed 42)

| Metric | Legacy Baseline | This Stack ($SE(2)$ Fusion) |
|---|---|---|
| Position RMSE | `0.3509 m` | **`0.1188 m`** |
| Overall Recall | `48.99%` | **`96.84%`** |
| Long-Range Recall (15–35 m) | `0.00%` | **`93.06%`** |
| Semantic Accuracy | `79.97%` | **`99.58%`** |
| Track ID Switches | `313` | **`0`** |

### CPU Step Latency

| Workload | Prior Unvectorized | Python / NumPy / ORT | C++20 `fusion_fast_kernels` |
|---|---|---|---|
| **Ouster OS1-64** (131,072 pts/scan + EKF) | `21.79 ms` | `8.76 ms` (NumPy) | **`5.70 ms`** |
| **Velodyne VLP-16** (16-beam + EKF) | `15.72 ms` | **`6.24 ms`** (NumPy) | — |
| **Synthetic Autocross** (80 frames) | — | `8.91 ms` mean, `19.42 ms` P99 (NumPy) | — |
| **Bundled `nano.onnx` (YOLO26n)** camera inference | — | `49.99 ms` mean (ONNX Runtime CPU) | — |

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
| `/ground_truth/cones` | `visualization_msgs/msg/MarkerArray` | 2 Hz | Ground-truth track cones in `odom` frame |

---

## Native Build & Testing

### 1. Build Workspace
```bash
source /opt/ros/humble/setup.bash
cd Fusion/ros2_ws
colcon build --packages-select fusion_interfaces fusion_core fusion_adapters
source install/setup.bash
```

### 2. Verify Pipeline Launches
```bash
# Synthetic test track simulation (zero hardware needed)
ros2 launch fusion_core synthetic_stack.launch.py

# Core fusion node standalone
ros2 launch fusion_core fusion.launch.py

# Full stack with camera and LiDAR adapters
ros2 launch fusion_adapters full_stack.launch.py

# Full stack with pre-bundled ONNX camera detector
ros2 launch fusion_adapters onnx_camera_stack.launch.py
```
