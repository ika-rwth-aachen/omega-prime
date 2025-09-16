ARG ROS_DISTRO=humble
ARG OMEGA_PRIME_VERSION=latest
ARG PERCEPTION_INTERFACES_REPO=https://github.com/ika-rwth-aachen/perception_interfaces.git
ARG PERCEPTION_INTERFACES_REF=

FROM osrf/ros:${ROS_DISTRO}-desktop

ARG ROS_DISTRO
ARG OMEGA_PRIME_VERSION
ARG PERCEPTION_INTERFACES_REPO
ARG PERCEPTION_INTERFACES_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROS_DISTRO=${ROS_DISTRO}

# System deps and ROS packages needed to build/use perception_msgs and utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-yaml \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-rosbag2-py \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
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

# Python deps: omega-prime from PyPI
WORKDIR /opt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir numpy==1.26.4 scipy==1.10.1 rosbags && \
    if [ "${OMEGA_PRIME_VERSION}" = "latest" ]; then \
      pip3 install omega-prime; \
    else \
      pip3 install "omega-prime==${OMEGA_PRIME_VERSION}"; \
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
