ARG ROS_DISTRO=humble
ARG OMEGA_PRIME_VERSION=latest
ARG PERCEPTION_INTERFACES_REPO=https://github.com/ika-rwth-aachen/perception_interfaces.git
ARG PERCEPTION_INTERFACES_REF=

FROM osrf/ros:${ROS_DISTRO}-ros-base

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
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Workspace for ROS Python messages and utils
WORKDIR /opt/ws
RUN mkdir -p /opt/ws/src

# Clone perception_interfaces and optionally checkout a specific ref (commit/branch/tag)
SHELL ["/bin/bash", "-c"]
RUN git clone ${PERCEPTION_INTERFACES_REPO} /opt/ws/src/perception_interfaces && \
    cd /opt/ws/src/perception_interfaces && \
    if [ -n "${PERCEPTION_INTERFACES_REF}" ]; then git checkout ${PERCEPTION_INTERFACES_REF}; fi

# Build only the required packages
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build \
      --merge-install \
      --symlink-install \
      --packages-up-to \
        perception_msgs \
        perception_msgs_utils \
        tf2_perception_msgs

# Python deps: ros2-unbag CLI, omega-prime from PyPI, tf-transformations
WORKDIR /opt
RUN pip3 install --upgrade pip && \
    pip3 install ros2-unbag tf-transformations && \
    if [ "${OMEGA_PRIME_VERSION}" = "latest" ]; then \
      pip3 install omega-prime; \
    else \
      pip3 install "omega-prime==${OMEGA_PRIME_VERSION}"; \
    fi

# Include the export routine inside the image for --use-routine without mounts
# Build context is repo root; copy from this repo path
RUN mkdir -p /opt/routines
COPY omega-prime/tools/ros2_unbag/object_list_to_omega_prime.py /opt/routines/object_list_to_omega_prime.py

# Simple wrapper to run exports for all bags in /data to /out
COPY omega-prime/tools/ros2_unbag/unbag-omega-prime.sh /usr/local/bin/unbag-omega-prime
RUN chmod +x /usr/local/bin/unbag-omega-prime

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
CMD ["unbag-omega-prime"]
