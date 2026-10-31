#!/bin/bash
set -e

# Source ROS 2 base installation
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source local workspace build
if [ -f "/workspace/Fusion/ros2_ws/install/setup.bash" ]; then
    source "/workspace/Fusion/ros2_ws/install/setup.bash"
elif [ -f "./install/setup.bash" ]; then
    source "./install/setup.bash"
fi

exec "$@"
