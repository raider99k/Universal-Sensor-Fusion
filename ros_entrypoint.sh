#!/bin/bash
set -e

# Source ROS 2 base installation
if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
    for _distro in humble jazzy rolling; do
        if [ -f "/opt/ros/${_distro}/setup.bash" ]; then
            source "/opt/ros/${_distro}/setup.bash"
            break
        fi
    done
fi

# Source local workspace build
if [ -f "/workspace/Fusion/ros2_ws/install/setup.bash" ]; then
    source "/workspace/Fusion/ros2_ws/install/setup.bash"
elif [ -f "./install/setup.bash" ]; then
    source "./install/setup.bash"
fi

exec "$@"
