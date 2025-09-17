ARG OMEGA_PRIME_VERSION=latest
ARG PERCEPTION_INTERFACES_REPO=https://github.com/ika-rwth-aachen/perception_interfaces.git
ARG PERCEPTION_INTERFACES_REF=

FROM osrf/ros:jazzy-desktop

ARG OMEGA_PRIME_VERSION
ARG PERCEPTION_INTERFACES_REPO
ARG PERCEPTION_INTERFACES_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROS_DISTRO=jazzy

# System deps and ROS packages needed to build/use perception_msgs and utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-jazzy-rosidl-default-generators \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-tf-transformations \
    ros-jazzy-rosbag2-py \
    ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

# Workspace for ROS Python messages and utils
WORKDIR /opt/ws
RUN mkdir -p /opt/ws/src

# Fetch perception_interfaces from GitHub so the image can be built standalone.
SHELL ["/bin/bash", "-c"]
RUN git clone ${PERCEPTION_INTERFACES_REPO} /opt/ws/src/perception_interfaces && \
    if [ -n "${PERCEPTION_INTERFACES_REF}" ]; then \
      cd /opt/ws/src/perception_interfaces && \
      git checkout ${PERCEPTION_INTERFACES_REF}; \
    fi

# Build only the required packages
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build \
      --merge-install \
      --symlink-install \
      --packages-up-to \
        perception_msgs \
        perception_msgs_utils \
        tf2_perception_msgs

# Python deps: omega-prime from PyPI inside isolated venv
WORKDIR /opt
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade scipy pyyaml transforms3d && \
    if [ "${OMEGA_PRIME_VERSION}" = "latest" ]; then \
      python -m pip install omega-prime; \
    else \
      python -m pip install "omega-prime==${OMEGA_PRIME_VERSION}"; \
    fi

# Include the converter inside the image
RUN mkdir -p /opt/ros2_conversion
COPY omega-prime/tools/ros2_conversion/object_list_to_omega_prime.py /opt/ros2_conversion/object_list_to_omega_prime.py

# Convenience entrypoint to ensure ROS env is sourced
ADD <<'EOS' /ros_entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
if [ -f /opt/ws/install/setup.bash ]; then
  source /opt/ws/install/setup.bash
fi
exec "$@"
EOS
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "/opt/ros2_conversion/object_list_to_omega_prime.py"]
