#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Execute the command passed to docker run
exec "$@"