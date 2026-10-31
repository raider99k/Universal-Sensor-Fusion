FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system utilities, build tools, ROS 2 standard packages, and foxglove bridge (CPU-only, zero GPU dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    git \
    python3-colcon-common-extensions \
    python3-pip \
    python3-pytest \
    python3-rosdep \
    ros-humble-diagnostic-msgs \
    ros-humble-foxglove-bridge \
    ros-humble-geometry-msgs \
    ros-humble-launch \
    ros-humble-launch-ros \
    ros-humble-nav-msgs \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-sensor-msgs-py \
    ros-humble-std-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-vision-msgs \
    ros-humble-visualization-msgs \
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

# Build ROS 2 workspace
WORKDIR /workspace/Fusion/ros2_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --packages-select fusion_interfaces fusion_core fusion_adapters --symlink-install"

# Expose Foxglove Studio WebSocket bridge port
EXPOSE 8765

WORKDIR /workspace
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "fusion_core", "synthetic_stack.launch.py"]
