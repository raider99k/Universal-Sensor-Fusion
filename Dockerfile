ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base
ARG ROS_DISTRO=humble

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system utilities, build tools, ROS 2 standard packages, and foxglove bridge (CPU-only, zero GPU dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    git \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-pytest \
    python3-rosdep \
    python3-scipy \
    ros-${ROS_DISTRO}-diagnostic-msgs \
    ros-${ROS_DISTRO}-foxglove-bridge \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-launch \
    ros-${ROS_DISTRO}-launch-ros \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-rclcpp-components \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-sensor-msgs-py \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-visualization-msgs \
    && (pip3 install --no-cache-dir --break-system-packages onnxruntime || pip3 install --no-cache-dir onnxruntime) \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Copy workspace source code
COPY Fusion/ros2_ws /workspace/Fusion/ros2_ws
COPY config /workspace/config
COPY ci /workspace/ci

# Copy entrypoint
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Build ROS 2 workspace (including C++20 kernels and composable nodes)
WORKDIR /workspace/Fusion/ros2_ws
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --packages-select fusion_interfaces fusion_core_cpp fusion_core fusion_adapters --symlink-install"

# Expose Foxglove Studio WebSocket bridge port
EXPOSE 8765

WORKDIR /workspace
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "fusion_core", "synthetic_stack.launch.py"]
