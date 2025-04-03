FROM ros:humble-ros-base

# Set working directory
WORKDIR /ros2_ws

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY examples/common /ros2_ws/src/neuracore/examples/common
COPY examples/ros_example /ros2_ws/src/ros_example
COPY examples/ros_example/entrypoint.sh /

RUN . /opt/ros/humble/setup.sh && \
    cd /ros2_ws && \
    colcon build --symlink-install --packages-select ros_example && \
    . install/setup.sh && \
    apt-get update && rosdep install --from-paths src --ignore-src -r -y

RUN pip3 install "neuracore[examples] @ git+https://git@github.com/Neuraco/neuracore.git@v2dev"
# Patch numpy for ROS
RUN pip3 install numpy==1.21.5 

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "ros_example", "data_collection.py"]
