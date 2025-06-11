FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
# RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
#     && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && apt-get -y install \
    git-lfs \
    libglvnd-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libosmesa6 \
    libglfw3 \
    patchelf \
    cmake \
    libx11-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglu1-mesa-dev \
    libxxf86vm-dev \
    xorg-dev \
    mesa-utils \
    libegl1-mesa \
    libglx-mesa0 \
    libgles2-mesa \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-opengl \
    default-jre \
    default-jdk


    
# Download and install MuJoCo
RUN mkdir -p /usr/local/mujoco && \
    curl -L -o mujoco.tar.gz https://github.com/google-deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz && \
    tar -xvzf mujoco.tar.gz -C /usr/local/mujoco && \
    rm mujoco.tar.gz

# Set environment variables
ENV MUJOCO_PATH=/usr/local/mujoco/mujoco-2.3.7
ENV LD_LIBRARY_PATH=$MUJOCO_PATH/bin:$LD_LIBRARY_PATH
ENV MUJOCO_PLUGIN_PATH=$MUJOCO_PATH/plugin
RUN mkdir -p $MUJOCO_PLUGIN_PATH

ENV DISPLAY=""
ENV NEURACORE_API_URL="http://localhost:8000/api"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

ENV MUJOCO_GL="osmesa"
ENV PYOPENGL_PLATFORM="osmesa"
ENV XDG_SESSION_TYPE="headless"