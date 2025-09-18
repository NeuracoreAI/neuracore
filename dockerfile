FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip mesa-utils && \
    rm -rf /var/lib/apt/lists/*