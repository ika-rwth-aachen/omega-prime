# ROS 2 Bag -> omega-prime Docker Image

This image bundles ROS 2, rosbag2 Python bindings, omega-prime (via PyPI), and builds perception_interfaces (messages + Python utils) from GitHub so you can export ObjectList topics to omega-prime MCAP using the built-in converter.

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
    -f omega-prime/tools/ros2_conversion/Dockerfile .
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
- The converter scans `/data` (or `OP_DATA`) for rosbag2 directories containing a `metadata.yaml` and writes one omega-prime `.mcap` per bag into `/out` (or `OP_OUT`).
- Provide `OP_ODR` (or `--map`) to embed an OpenDRIVE map in the outputs.
- For large bags, write to a mounted folder and ensure sufficient RAM.

## Advanced
- Env vars / CLI flags:
  - `OP_DATA` / `--data-dir` (default `/data`)
  - `OP_OUT` / `--output-dir` (default `/out`)
  - `OP_TOPIC` / `--topic` (required)
  - `OP_ODR` / `--map`
  - `OP_VALIDATE` / `--validate`
  - `--bag` to process explicit bag directories in addition to auto-discovery

## OpenDRIVE Map Integration

### During export (recommended)
- Place your `.xodr` file under the mounted `/data` (or another mounted path) and set `-e OP_ODR=/data/your_map.xodr`.
- The export routine embeds the map in each generated omega-prime `.mcap`.

### Notes
- If `OP_ODR` is not set and you don’t post-process, outputs won’t include a map.
- Map parsing uses a default geometry sampling step size of 0.01 m.
