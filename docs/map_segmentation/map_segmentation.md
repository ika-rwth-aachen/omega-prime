# Map Segmentation (omega_prime.map_segmentation)

This document describes the MapSegmentation system used to group map lanes into semantic segments (junctions and non-junction road segments), attach traffic lights, and offer utilities for plotting and trajectory-to-segment mapping.

It covers:
- Architecture and class hierarchy
- What the classes do and the data they expect
- The step-by-step processing pipeline
- Algorithms and thresholds used
- Public API surface and typical usage
- Assumptions, limitations, and edge cases


## Overview

The map segmentation system uses an **abstract base class pattern** to support different map types (e.g., OSI full maps, OSI centerline-only maps). The core abstractions are:

- **`MapSegmentation` (ABC)**: Base class defining the segmentation pipeline and common operations
- **`Segment` (ABC)**: Base class for map segments (junctions, connections, etc.)
- **`MapOsiCenterlineSegmentation`**: Concrete implementation for OSI centerline maps
- **`SegmentOsiCenterline`**: Concrete segment implementation for OSI centerline maps
- **`Intersection`** and **`ConnectionSegment`**: Specialized segment types

### Segment Types

MapSegmentation clusters lanes into segments of type:
- **JUNCTION** (intersections): groups of lanes whose buffered geometries intersect
- **STRAIGHT** (connection segments): connected components of lanes outside junctions
- Other types exist in the enum but are not currently assigned by the code (e.g., RAMP_ON, RAMP_OFF, ROUNDABOUT)

Each segment has a polygon generated from lane centerlines (concave hull) and may have traffic lights attached. The system also:
- Marks lanes as `on_intersection` or `is_approaching` (predecessors of intersection lanes)
- Assigns traffic lights to nearest lanes
- Optionally updates lane `road_id` to the segment ID
- Splits trajectories into segment-wise chunks
- Provides plotting utilities for quick inspection


## Architecture

### Abstract Base Classes

The system is built on two abstract base classes in `mapsegment.py`:

#### `Segment` (ABC)
Base class for all segment types. Defines abstract methods that must be implemented by concrete segment classes:
- `_get_lane_id(lane)`: Extract lane ID from a lane object
- `_get_lane_geometry(lane)`: Extract geometry (LineString) from a lane object  
- `set_trafficlight()`: Set traffic lights for the segment

Common functionality provided:
- Polygon computation with caching (concave hull with fallback to convex hull)
- Lane management (`add_lane`, lane IDs tracking)
- Center point calculation
- Time interval analysis for road users on segment

#### `MapSegmentation` (ABC)
Base class for map segmentation implementations. Defines abstract methods for map-type-specific operations.


Common functionality provided:
- Lane dictionary creation
- Successor/predecessor mapping
- Validation that all lanes are assigned to segments

### Concrete Implementations for OSI Centerline Maps

In `maposicenterlinesegmentation.py`:

#### `SegmentOsiCenterline`
Concrete implementation of `Segment` for OSI centerline maps.

#### `MapOsiCenterlineSegmentation`
Concrete implementation of `MapSegmentation` for OSI centerline maps. Implements all abstract methods and provides the full segmentation pipeline including:
- Parallel lane detection using spatial indexing
- Intersection detection via graph analysis
- Connection segment identification
- Road ID updates
- Trajectory segmentation

#### `Intersection` and `ConnectionSegment`
Specialized segment classes extending `SegmentOsiCenterline`:
- `Intersection`: Represents junction segments (type = JUNCTION)
- `ConnectionSegment`: Represents connection segments (type = STRAIGHT), also tracks which intersections it connects


## Inputs and Data Model

The constructor expects a `recording` object with at least:
- `recording.map`: provides `map.lanes` (iterable of lane objects)
- `recording.traffic_light_states`: mapping of traffic light states over time (used to discover static traffic lights)

### Lane Object Requirements (for OSI Centerline implementation)

Lane objects are assumed to provide:
- `lane.idx`: a key/index object with fields: `.lane_id` (unique lane identifier) and `.road_id` (road grouping id)
- `lane.centerline`: a shapely LineString geometry in meters
- `lane.successor_ids` and `lane.predecessor_ids`: each item is either a lane ID or an object with `.lane_id`
- Optional attributes set/used by MapSegmentation: `lane.on_intersection`, `lane.is_approaching`, and `lane.trafficlight`

**Note**: Different map types may have different lane object requirements. Concrete implementations define how to extract this information via the abstract methods.

Coordinate system and units:
- All XY coordinates are in meters in a consistent local frame.


## Configuration Parameters

Constructor parameters (with defaults) for `MapOsiCenterlineSegmentation`:
- `lane_buffer` (float, default 0.3 m): buffer used to detect intersecting lanes (geometric proximity)
- `intersection_overlap_buffer` (float, default 1.0 m): buffer used to determine when two intersections should be merged
- `concave_hull_ratio` (float, default 0.3): ratio parameter for concave hull polygon generation (range 0-1, where 0 is convex hull and 1 is maximally concave)

The base `MapSegmentation` class also accepts:
- `concave_hull_ratio` (float, default 0.3): passed through to all segment instances for polygon generation

Internal thresholds/constants:
- Parallel lane search radius: 10 m (buffer around a lane when searching for parallels)
- Parallel lane angle threshold: < 10 degrees between centerline directions
- Trajectory re-segmentation buffer around intersections: 5 m
- Combine isolated connections connecting the same single intersection if polygon distance < 5 m


## Segments and Polygons

Each segment is represented by a `Segment` base class with the following structure:
- Holds `lanes`, `lane_ids`, `trafficlights`, `idx`, `type`
- Computes a polygon once and caches it; polygon is a concave hull of the union of lane centerlines
- `create_segment_polygon()` / `update_polygon()` maintain cache consistency
- Uses abstract methods `_get_lane_id()` and `_get_lane_geometry()` to extract lane-specific information

Specializations:
- `Intersection(SegmentOsiCenterline)`: `type = JUNCTION`
- `ConnectionSegment(SegmentOsiCenterline)`: `type = STRAIGHT`; also tracks `intersection_idxs` it connects

Polygon generation uses `concave_hull` with ratio `0.3`. The polygon is re-built when lanes change. If concave hull fails, it falls back to convex hull.

**Extensibility**: To support a new map type, create:
1. A concrete `Segment` subclass implementing the three abstract methods
2. A concrete `MapSegmentation` subclass implementing the nine abstract methods
3. Optionally, specialized segment types (like `Intersection` and `ConnectionSegment` for OSI centerline maps)


## Algorithms and Key Details

- Parallel lane detection
  - For each lane, build a 10 m buffer around its centerline, query STRtree of original centerlines, then compare direction vectors (angle < 10°) to mark as parallel.

- Intersecting lanes (potential junction membership)
  - For each lane, create a `lane_buffer`-sized buffer and query STRtree for nearby lanes. Exclude successors, predecessors, and parallels. If candidate centerline intersects the buffer, it is considered intersecting.

- Intersection detection and merging
  - Build a graph where nodes are lanes and edges indicate geometric intersection; connected components form `Intersection` objects.
  - Merge intersections whose polygons overlap when buffered by `intersection_overlap_buffer` to avoid fragmented junctions.
  - Add non-intersecting lanes that are contained within the buffered intersection polygon.

- Connection segments
  - For lanes not assigned to intersections, build a graph linking successors and predecessors; connected components form `ConnectionSegment` objects.
  - If multiple connection components connect the same pair(s) of intersections, merge them. If they relate to a single intersection, merge when polygons are within 5 m.

- Traffic lights
  - From `recording.traffic_light_states`, unique traffic lights are collected and assigned to the nearest lane (via STRtree nearest query on centerlines).

- Trajectory segmentation
  - `trajectory_segment_detection(trajectory)` maps a timestamped trajectory (n x 3: frame/time, x, y) to lane IDs via `Locator.xys2sts`, then to segment indices, and refines intersection assignment using buffered intersection polygons. Returns a list of tuples (trajectory_chunk, segment_object).


## Public API Summary

### Base Classes (for extension)

- **`MapSegmentation(recording)` (ABC)**
  - Abstract base class for map segmentation implementations.
  - Subclasses must implement 9 abstract methods for map-type-specific operations.
  - Provides common functionality: lane dictionaries, successor/predecessor mapping, validation.

- **`Segment(lanes, idx=None)` (ABC)**
  - Abstract base class for segment implementations.
  - Subclasses must implement 3 abstract methods for lane-specific operations.
  - Provides polygon computation, caching, lane management, and road user analysis.

### Concrete Implementation for OSI Centerline Maps

- **`MapOsiCenterlineSegmentation(recording, lane_buffer=0.3, intersection_overlap_buffer=1.0, concave_hull_ratio=0.3)`**
  - Create a segmentation object from a recording/map with OSI centerline lanes.
  - Implements all abstract methods from `MapSegmentation`.
  - `concave_hull_ratio`: Controls the shape of segment polygons (0=convex, 1=maximally concave).

- **`init_intersections()`**
  - Run the full pipeline to populate intersections, connections, polygons, lane flags, traffic light mapping, and segment indices.

- **`trajectory_segment_detection(trajectory: np.ndarray) -> list[(np.ndarray, Segment)]`**
  - Split a time-ordered trajectory into segment-wise chunks; each chunk is an array of (time, x, y).

- **`plot(output_plot=None, trajectory=None, plot_lane_ids=False, plot_intersection_polygons=False, plot_connection_polygons=False)`**
  - Plot lanes and segments; optionally render polygons and annotate IDs. If `output_plot` is a Path, saves a PDF.

- **`plot_intersections(output_plot)`**
  - Plot each intersection/connection into separate PDFs.

### Segment Classes

- **`SegmentOsiCenterline(lanes, idx=None, concave_hull_ratio=0.3)`**
  - Concrete segment implementation for OSI centerline maps.

- **`Intersection(lanes, idx=None, concave_hull_ratio=0.3)`**
  - Specialized segment for junctions (extends `SegmentOsiCenterline`).

- **`ConnectionSegment(lanes, idx=None, concave_hull_ratio=0.3)`**
  - Specialized segment for connections (extends `SegmentOsiCenterline`).



## Assumptions and Limitations

- **Geometry and units**
  - Coordinates are meters in a consistent local frame; lane centerlines are valid shapely LineStrings.
  - The concave hull at ratio 0.3 yields sensible junction polygons; highly irregular geometries may need tuning.

- **Topology**
  - Successor/predecessor IDs are consistent with lane IDs (or objects exposing `.lane_id`).
  - Parallel lanes share a similar direction; 10° threshold may not capture very slight divergence or reversible lanes.

- **Spatial thresholds**
  - `lane_buffer` (default 0.3 m) controls when lanes are deemed intersecting; maps with larger lane spacing may need higher values.
  - `intersection_overlap_buffer` (default 1.0 m) controls intersection merging; too small can fragment, too large can over-merge.

- **Traffic lights**
  - Assignment is nearest-centerline based. Complex lane-level signal mapping is out of scope.

- **Road ID updates**
  - `update_road_ids()` reassigns `lane.idx.road_id` to the segment index, modifies `lane.idx` and updates dict keys and successor/predecessor references. If external code relies on original road IDs or lane indices as dict keys, ensure compatibility before enabling this behavior downstream.

- **Performance**
  - Uses STRtree spatial indices for near-neighbor queries. Still, very large maps may require tuning buffer sizes.

- **Robustness and warnings**
  - If a lane is encountered in more than one segment, a warning is printed and the assignment is not duplicated.
  - If no lanes or traffic lights exist, the pipeline degrades gracefully (empty structures).

- **Map type specific implementations**
  - Currently, only OSI centerline maps are fully implemented (`MapOsiCenterlineSegmentation`).
  - To support other map types, implement the abstract methods in new subclasses of `MapSegmentation` and `Segment`.


## Edge Cases to Consider

- Empty map (no lanes): all data structures remain empty; plotting produces a blank map.
- Degenerate or extremely short centerlines: parallel detection and hull computation can be unstable; ensure valid geometries.
- Very dense lane networks: increase `lane_buffer` and `intersection_overlap_buffer` prudently to avoid over-fragmentation.
- Miswired predecessors/successors: can impact approaching-lane marking and connection grouping.
- Trajectories straddling polygon boundaries: buffered checks attempt to align intersection membership; adjust `buffer=5` in `trajectory_segment_detection` if needed.


## Extensibility Notes

- **Supporting new map types**
  - Create a concrete `Segment` subclass implementing:
    - `_get_lane_id(lane)`: Extract lane ID
    - `_get_lane_geometry(lane)`: Extract lane geometry (LineString)
    - `set_trafficlight()`: Set traffic lights for the segment
  - Create a concrete `MapSegmentation` subclass implementing:
    - `_get_lane_id(lane)`: Extract lane ID
    - `_get_lane_centerline(lane)`: Extract centerline
    - `_get_lane_successors(lane)`: Extract successor IDs
    - `_get_lane_predecessors(lane)`: Extract predecessor IDs
    - `_has_traffic_light(lane)`: Check for traffic light
    - `_get_traffic_light(lane)`: Get traffic light object
    - `_set_lane_on_intersection(lane, value)`: Set intersection flag
    - `_set_lane_is_approaching(lane, value)`: Set approaching flag
    - `_get_lane_on_intersection(lane)`: Get intersection status
  - Optionally override methods like `init_intersections()` if the pipeline needs customization.

- **Additional segment types**
  - The enum `MapSegmentType` includes non-used types (ramp_on, ramp_off, roundabout). You can add classifiers to set `segment.type` based on topology/geometry.

- **Alternative polygon generation**
  - Swap concave hull for convex hull or buffered union, or adapt concave hull ratio per segment size by overriding `_compute_segment_polygon()` in your `Segment` subclass.

- **Better traffic light mapping**
  - Replace nearest-lane heuristic with spatial joins to stop lines or explicit topology by overriding `set_lane_trafficlights()`.

- **Visualization**
  - Add tiled basemaps or interactive inspection leveraging geopandas/folium if appropriate by extending the `plot()` methods.
