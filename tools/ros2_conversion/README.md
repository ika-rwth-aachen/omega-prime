# ROS 2 Bag -> omega-prime Docker Image

This image bundles ROS 2 Jazzy, its rosbag2 Python bindings, omega-prime (via PyPI), and builds perception_interfaces (messages + Python utils) from GitHub so you can export ObjectList topics to omega-prime MCAP using the built-in converter.

## Build Args
- `OMEGA_PRIME_VERSION` (default `latest`): PyPI version to install; use `latest` for newest
- `PERCEPTION_INTERFACES_REPO` (default GitHub repo)
- `PERCEPTION_INTERFACES_REF` (optional): commit/branch/tag; if unset, uses the repo’s default branch

## Build
```bash
# Run from omega-prime root directory
docker build -t omega-op-unbag \
    --build-arg OMEGA_PRIME_VERSION=latest \
    --build-arg PERCEPTION_INTERFACES_REF=<commit-or-branch> \
    -f tools/ros2_conversion/Dockerfile .
```

## Run (simple)
- Mount your bag directory to `/data` and an output directory to `/out`.
- Set the topic via `OP_TOPIC` (ObjectList topic); the container runs the export automatically.

### Example:
```bash
docker run --rm -it \
    -e OP_TOPIC=/your/object_list_topic \
    -v /path/to/bags:/data:ro \
    -v /path/to/map.xodr:/map/map.xodr:ro \
    -v "$PWD"/out:/out \
    omega-op-unbag
```

## Notes
- The image builds and installs `perception_interfaces` packages needed for Python APIs and messages (`perception_msgs`, `perception_msgs_utils`, `tf2_perception_msgs`).
- The converter scans `/data` (or `OP_DATA`) for rosbag2 directories containing a `metadata.yaml` and writes one omega-prime `.mcap` per bag into `/out` (or `OP_OUT`).
- For large bags, write to a mounted folder and ensure sufficient RAM.

## Advanced
- Env vars / CLI flags:
  - `OP_DATA` / `--data-dir` (default `/data`)
  - `OP_OUT` / `--output-dir` (default `/out`)
  - `OP_TOPIC` / `--topic` (required)
  - `OP_VALIDATE` / `--validate`
  - `--bag` to process explicit bag directories in addition to auto-discovery

## OpenDRIVE Map Integration

### During export (recommended)
- Place your `.xodr` file under the mounted `/data/map.xodr`
- The export routine embeds the map in each generated omega-prime `.mcap`.

### Notes
- If `/data/map.xodr` does not exist, outputs won’t include a map.
- Map parsing uses a default geometry sampling step size of 0.01 m.
