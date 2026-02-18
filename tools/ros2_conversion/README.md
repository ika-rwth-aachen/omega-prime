# ROS 2 Bag -> omega-prime Docker Image

This image bundles ROS 2 Jazzy, its rosbag2 Python bindings, omega-prime (via PyPI), and builds perception_interfaces (messages + Python utils) from GitHub so you can export ObjectList topics to omega-prime MCAP using the built-in converter.

## Build Args
- `OMEGA_PRIME_VERSION` (default `latest`): PyPI version to install; use `latest` for newest
- `PERCEPTION_INTERFACES_REPO` (default GitHub repo)
- `PERCEPTION_INTERFACES_REF` (optional): commit/branch/tag; if unset, uses the repo’s default branch

## Build
```bash
# Run from omega-prime root directory
docker build -t ros2omegaprime \
    --build-arg OMEGA_PRIME_VERSION=latest \
    --build-arg PERCEPTION_INTERFACES_REF=<commit-or-branch> \
    -f tools/ros2_conversion/Dockerfile .
```

## Run
- Mount your bag directory to `/data` and an output directory to `/out`.
- Set the topic via `OP_TOPIC` (ObjectList topic); the container runs the export automatically.

### Example:
```bash
docker run --rm -it \
    -e OP_TOPIC=</your/object_list_topic> \
    -v <path/to/bags>:/data:ro \
    -v </path/to/map.xodr>:/map/map.xodr:ro \
    -v "$PWD"/out:/out \
    ros2omegaprime
```

## Projections and Fixed Frame
- The converter reads `/tf` + `/tf_static` and resolves each ObjectList message frame against the configured `fixed_frame`.
- The `fixed_frame` should be the georeferenced top-level ROS coordinate frame (TF root) of your setup, for example the global UTM/world frame. When `fixed_frame` is map, the map must be parsed and the projection string is the one of the map.
- These transforms are stored in omega-prime as per-timestamp `ProjectionOffset` metadata.
- The fixed frame is converted to an EPSG projection string and written as `projections["proj_string"]`.
- Supported fixed frame values are currently:
  - `utm_30N` -> `EPSG:32630`
  - `utm_31N` -> `EPSG:32631`
  - `utm_32N` -> `EPSG:32632`
  - `utm_33N` -> `EPSG:32633`
- If you want to add a new EPSG definition, look up the corresponding `utm` zone (for example using [this](https://de.wikipedia.org/wiki/UTM-Koordinatensystem#/media/Datei:Utm-zones.jpg) map) of your fixed frame. With that information, you can find the correct EPSG code [here](https://epsg.io/).

## Notes
- The image builds and installs `perception_interfaces` packages needed for Python APIs and messages (`perception_msgs`, `perception_msgs_utils`, `tf2_perception_msgs`).
- The converter scans `/data` for rosbag2 directories containing a `metadata.yaml` and writes one omega-prime `.mcap` per bag into `/out` per default.
- For large bags ensure sufficient RAM.

## Advanced
- Env vars / CLI flags:
  - `OP_DATA` / `--data-dir` (default `/data`)
  - `OP_OUT` / `--output-dir` (default `/out`)
  - `OP_TOPIC` / `--topic` (required)
  - `OP_VALIDATE` / `--validate`
  - `OP_FIXED_FRAME` / `--fixed_frame` (default `utm_32N`)
  - `--bag` to process explicit bag directories in addition to auto-discovery
  - `--timeout` Set a timeout that prints a warning if the same object ID appears again after `timeout` seconds.

## OpenDRIVE Map Integration

### During export (recommended)
- Place your `.xodr` file under the mounted `/map/map.xodr`
- The export routine embeds the map in each generated omega-prime `.mcap`.

### Notes
- If `/map/map.xodr` does not exist, outputs won’t include a map.
- Map parsing uses a default geometry sampling step size of 0.01 m.
