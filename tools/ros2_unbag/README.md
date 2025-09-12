# ROS 2 + ros2_unbag Docker Image for omega-prime Export

This image bundles ROS 2, ros2_unbag, omega-prime (via PyPI), and builds perception_interfaces (messages + Python utils) from GitHub, so you can export ObjectList topics to omega-prime MCAP using the built-in routine.

## Build Args
- `ROS_DISTRO` (default `humble`)
- `OMEGA_PRIME_VERSION` (default `latest`): PyPI version to install; use `latest` for newest
- `PERCEPTION_INTERFACES_REPO` (default GitHub repo)
- `PERCEPTION_INTERFACES_REF` (optional): commit/branch/tag; if unset, uses the repo’s default branch

## Build
```bash
docker build -t omega-op-unbag \
    --build-arg ROS_DISTRO=humble \
    --build-arg OMEGA_PRIME_VERSION=latest \
    --build-arg PERCEPTION_INTERFACES_REF=<commit-or-branch> \
-f omega-prime/tools/ros2_unbag/Dockerfile .
```

## Run (simple)
- Mount your bag directory to `/data` and an output directory to `/out`.
- Set the topic via `OP_TOPIC` (ObjectList topic); the container runs the export automatically.

### Example:
```bash
docker run --rm -it \
    -e OP_TOPIC=/your/object_list_topic \
    -e OP_ODR=/data/map.xodr \
    -v /path/to/bags:/data \
    -v "$PWD"/out:/out \
    omega-op-unbag
```

## Notes
- The image builds and installs `perception_interfaces` packages needed for Python APIs and messages (`perception_msgs`, `perception_msgs_utils`, `tf2_perception_msgs`).
- The routine writes a single omega-prime `.mcap` without OpenDRIVE. If you later want to embed a map, extend the routine to set `Recording.map`.
- For large bags, write to a mounted folder and ensure sufficient RAM.

## Advanced
- Env vars:
  - `OP_DATA` (default `/data`), `OP_OUT` (default `/out`)
  - `OP_TOPIC` (required): ObjectList topic
  - `OP_GLOB` (default `"*.mcap *.db3"`): filename patterns in `/data`
  - `OP_FORMAT` (default `omega-prime/MCAP`)
  - `OP_ROUTINE` (default `/opt/routines/object_list_to_omega_prime.py`)
  - `OP_EXTRA_ARGS`: forwarded to `ros2 unbag`
  - `OP_ODR`: optional path to a `.xodr` inside the container (e.g., `/data/map.xodr`) to embed the map in the output MCAP

## OpenDRIVE Map Integration

### During export (recommended)
- Place your `.xodr` file under the mounted `/data` (or another mounted path) and set `-e OP_ODR=/data/your_map.xodr`.
- The export routine embeds the map in each generated omega-prime `.mcap`.

### Notes
- If `OP_ODR` is not set and you don’t post-process, outputs won’t include a map.
- Map parsing uses a default geometry sampling step size of 0.01 m.
