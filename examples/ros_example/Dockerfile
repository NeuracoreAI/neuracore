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

# Once everything finalised, we can do this
# COPY . /ros2_ws/src/ros_example/
# RUN pip3 install --no-cache-dir neuracore[examples]

# ======
COPY examples/ros_example /ros2_ws/src/ros_example
COPY examples/ros_example/entrypoint.sh /
# =====

RUN . /opt/ros/humble/setup.sh && \
    cd /ros2_ws && \
    colcon build --symlink-install --packages-select ros_example && \
    . install/setup.sh && \
    apt-get update && rosdep install --from-paths src --ignore-src -r -y

# ====
COPY . /ros2_ws/src/neuracore
RUN cd /ros2_ws/src/neuracore && pip install .[examples]
# Patch for ros
RUN pip3 install numpy==1.21.5
# ====

# COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["xvfb-run ros2 launch ros_example data_collection.py"]

# docker build --file examples/ros_example/Dockerfile -t  ros_example:latest .
# docker run -it --rm ros_example:latest 