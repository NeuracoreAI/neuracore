#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

Xvfb :99 &
XVFB_PROC=$!
sleep 1
export DISPLAY=:99
"$@"
kill $XVFB_PROC
