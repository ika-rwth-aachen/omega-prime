# Map Segmentation (omega_prime.mapsegment)

This document describes the map segmentation system used to group map lanes into semantic segments, attach available traffic-light information, plot segmentation results, and split trajectories into segment-wise chunks.

It covers:
- Architecture and class hierarchy
- OSI centerline and OpenDRIVE segmentation behavior
- Data model expectations
- Algorithms and thresholds
- Public API surface
- Assumptions, limitations, and edge cases


## Overview

The map segmentation system uses an abstract base class pattern to support multiple map types. The currently implemented segmentation classes are:

- `MapSegmentation`: shared base class for segmentation implementations
- `Segment`: shared base class for segment objects
- `MapOsiCenterlineSegmentation`: segmentation for `MapOsiCenterline`
- `MapODRSegmentation`: segmentation for `MapOdr` / OpenDRIVE maps
- `SegmentOsiCenterline`, `Intersection`, `ConnectionSegment`: OSI centerline segment classes
- `SegmentOdr`, `IntersectionOdr`, `ConnectionSegmentOdr`: OpenDRIVE segment classes

Segments are currently assigned as:
- `MapSegmentType.JUNCTION`: intersection or OpenDRIVE junction segments
- `MapSegmentType.NO_JUNCTION`: non-junction connection segments

Other enum values exist, such as `RAMP_ON`, `RAMP_OFF`, and `ROUNDABOUT`, but are not currently assigned by these pipelines.

Each segment stores its lanes, segment index, segment type, and a cached polygon generated from lane centerlines. The shared base class also owns a `Locator` and provides lane-first trajectory-to-segment mapping for both OSI and ODR maps.


## Architecture

### `Segment`

`Segment` is the base class for all segment types. Concrete subclasses implement:
- `_get_lane_id(lane)`: extract the map-type-specific lane ID
- `_get_lane_geometry(lane)`: extract the lane geometry used for polygon generation
- `set_trafficlight()`: attach traffic-light information when available

Shared segment behavior includes:
- Polygon computation with caching
- Concave hull generation with convex hull fallback
- Lane management through `add_lane()`
- Center point calculation
- Road-user time-interval checks against the segment polygon
- Segment plotting with lane centerlines, lane labels, and optional polygon output semantics shared across map types

### `MapSegmentation`

`MapSegmentation` is the base class for map segmentation implementations. Concrete subclasses implement lane ID, centerline, successor, predecessor, traffic-light, and intersection-flag accessors.

Shared segmentation behavior includes:
- Storing `recording.map`, `map.lanes`, and common segment dictionaries
- Creating a `Locator` with `Locator.from_map(recording.map)`
- Building lane, successor, predecessor, and road-to-segment mappings
- Validating whether all lanes are assigned
- Splitting trajectories into segment-wise chunks with `trajectory_segment_detection()`
- Plotting full segmentation maps and per-segment intersection/connection plots through shared `plot()` and `plot_intersections()` implementations

### OSI Centerline Implementation

`MapOsiCenterlineSegmentation` builds intersections from lane geometry and topology:
- Detects parallel lanes with spatial indexing and direction comparison
- Detects intersecting lanes using buffered centerlines
- Builds graph components for intersections
- Merges nearby intersection components
- Finds connection segments from non-intersection lane topology
- Assigns traffic lights from `recording.traffic_light_states`
- Updates OSI lane `road_id` values to segment indices

`Intersection` and `ConnectionSegment` extend `SegmentOsiCenterline`.

### OpenDRIVE Implementation

`MapODRSegmentation` uses OpenDRIVE road metadata instead of graph-based intersection detection:
- Roads whose `junction` attribute is not `"-1"` are grouped into `IntersectionOdr` segments by junction ID
- Remaining roads are grouped into `ConnectionSegmentOdr` segments by road ID
- Segment indices are assigned sequentially from one shared namespace
- Original OpenDRIVE IDs are preserved on the segment as `odr_junction_id` or `odr_road_id`
- ODR lane IDs use the full `XodrLaneId` (`road_id`, `lane_id`, `section_id`) to avoid collisions

If ODR segment creation fails or any lane remains unassigned, `identify_segments()` raises a `RuntimeError` with details about failed segments and unassigned lanes.


## Inputs and Data Model

The constructor expects a `recording` object with:
- `recording.map`: a map with `map.lanes`
- `recording.traffic_light_states`: required by OSI centerline segmentation for traffic-light discovery

All XY coordinates are expected to be in meters in a consistent local frame.

### OSI Centerline Lane Requirements

OSI centerline lanes are expected to provide:
- `lane.idx.lane_id`: unique lane identifier used as the segmentation key
- `lane.idx.road_id`: road grouping ID, later updated to the segment index by `update_road_ids()`
- `lane.centerline`: shapely `LineString`
- `lane.successor_ids` and `lane.predecessor_ids`: lane IDs or objects exposing `.lane_id`
- Optional/assigned attributes: `lane.on_intersection`, `lane.is_approaching`, and `lane.trafficlight`

### OpenDRIVE Lane Requirements

OpenDRIVE lanes are expected to provide:
- `lane.idx`: full `XodrLaneId` namedtuple with `road_id`, `lane_id`, and `section_id`
- `lane.centerline`: shapely `LineString`
- `lane.successor_ids` and `lane.predecessor_ids`: full `XodrLaneId` values
- `lane.on_intersection`: derived from the lane classification

OpenDRIVE traffic-light mapping is currently a no-op in `SegmentOdr`.


## Configuration Parameters

### Shared Parameters

- `concave_hull_ratio` (default `0.3`): controls the tightness of the concave hull computed for each segment polygon. A value of `0.0` produces the tightest possible fit (approaching the input geometry); `1.0` degenerates to a convex hull. The default of `0.3` provides a good balance between tightly wrapping curved or branching lane geometry, which a convex hull would over-approximate, and robustness against noisy or very short centerlines.

### OSI Centerline Parameters

`MapOsiCenterlineSegmentation(recording, lane_buffer=0.3, intersection_overlap_buffer=1.0, concave_hull_ratio=0.3)`:
- `lane_buffer`: buffer used to detect intersecting lanes
- `intersection_overlap_buffer`: buffer used to merge nearby intersection components

Internal OSI thresholds:
- Parallel lane search radius: `10 m`
- Parallel lane angle threshold: `< 10 degrees`
- Isolated connection merge distance for a single bordering intersection: `5 m`

### OpenDRIVE Parameters

`MapODRSegmentation(recording, concave_hull_ratio=0.3)`:
- Uses only `concave_hull_ratio`
- Uses OpenDRIVE road `junction` attributes for segment grouping


## Segments and IDs

All segment classes hold:
- `lanes`
- `lane_ids`
- `trafficlights`
- `idx`
- `type`
- `polygon`

Segment polygons are concave hulls of lane centerlines using `concave_hull_ratio`; if concave hull generation fails, the code falls back to a convex hull.

A **concave hull** is used rather than a convex hull because road geometry is inherently non-convex:

- Lane centerlines within a segment can curve, run in parallel with gaps between them, or radiate outward from an intersection node. A convex hull would fill in all of that empty space, producing a polygon that extends well beyond the actual lane footprint.
- The polygon is used in `trajectory_segment_detection()` to verify that a located point actually lies within the segment. An over-extended convex hull would cause road users travelling on a neighbouring segment to be falsely assigned to the wrong segment.
- At multi-road intersections, the incoming and outgoing lanes spread outward, making the true segment footprint concave. Using a convex hull there would swallow territory belonging to the adjacent connection segments.

The convex hull fallback is retained for degenerate geometries (e.g. collinear or very short lanes) where Shapely's `concave_hull` raises a `GEOSException` or returns an empty geometry.

OSI behavior:
- Segment IDs are sequential.
- `update_road_ids()` rewrites OSI lane `idx.road_id` to the segment index.
- `lane_segment_dict` is keyed by bare `lane_id`.

ODR behavior:
- Segment IDs are sequential and unique across junction and road segments.
- Original OpenDRIVE road and junction IDs remain separate from `segment.idx`.
- `IntersectionOdr.odr_junction_id` stores the source junction ID.
- `ConnectionSegmentOdr.odr_road_id` stores the source road ID.
- `lane_segment_dict` is keyed by full `XodrLaneId`.


## Algorithms and Key Details

### OSI Centerline Pipeline

- Build lane, successor, and predecessor dictionaries.
- Detect parallel lanes using STRtree queries and centerline direction comparison.
- Detect intersecting lanes by querying buffered centerlines and excluding successors, predecessors, and parallels.
- Build a graph where intersecting lanes form edges; connected components become intersections.
- Merge overlapping or nearby intersection polygons.
- Add non-intersecting lanes contained within the expanded OSI intersection area.
- Build connection segments from connected components of non-intersection lanes.
- Assign traffic lights to nearest lanes using STRtree nearest queries.
- Assign sequential segment IDs and update lane `road_id` values.

### OpenDRIVE Pipeline

- Build lane, successor, and predecessor dictionaries.
- Build a `road_id -> junction_id` mapping from OpenDRIVE roads whose `junction` attribute is not `"-1"`.
- Group lanes on junction roads into `IntersectionOdr` segments by junction ID.
- Group all remaining lanes into `ConnectionSegmentOdr` segments by road ID.
- Assign sequential segment indices across all ODR segments.
- Build `segment_by_road_id` and `lane_segment_dict`.
- Raise `RuntimeError` if any segment failed to construct or any lane remains unassigned.

### Trajectory Segmentation

`trajectory_segment_detection(trajectory)` is implemented on `MapSegmentation` and shared by OSI and ODR.

Input:
- `trajectory` must be a `np.ndarray` with shape `(n, 3)`.
- Columns are `(frame_or_time, x, y)`.
- Empty `(0, 3)` trajectories return `[]`.
- Invalid shapes or non-NumPy inputs raise `ValueError`.

Behavior:
- Uses `self.locator.xys2sts(trajectory[:, 1:3])` to locate each point on a lane.
- Converts located lane IDs to the key used by `lane_segment_dict`.
- Looks up the assigned segment for each located lane.
- If the located lane has a polygon, verifies the point is covered by that polygon using `covers()`.
- Splits the trajectory whenever the resolved segment object changes.

Map-specific lane-key handling:
- OSI centerline segmentation converts located lane IDs to `.lane_id`.
- ODR segmentation uses the full located `XodrLaneId` directly.

Failure modes:
- Unknown located lane: `RuntimeError`
- Located lane has no assigned segment: `RuntimeError`
- Located lane cannot be resolved to a lane object: `RuntimeError`
- Point is outside the located lane polygon: `RuntimeError`

The trajectory method no longer uses buffered intersection segment polygons to reassign points near intersections; assignment is lane-first and map-type consistent.


## Public API Summary

### Shared Base API

- `MapSegmentation(recording, concave_hull_ratio=0.3)`
  - Base class for segmentation implementations.
  - Creates common dictionaries and a `Locator`.
  - Provides `trajectory_segment_detection()`, `build_segment_by_road_id()`, and `get_segment()`.

- `trajectory_segment_detection(trajectory: np.ndarray) -> list[tuple[np.ndarray, Segment]]`
  - Splits a time-ordered trajectory into segment-wise chunks.

- `plot(output_plot=None, trajectory=None, plot_lane_ids=False, plot_intersection_polygons=False, plot_connection_polygons=False)`
  - Shared map plot for all segmentation implementations.

- `plot_intersections(output_plot=None)`
  - Shared per-segment plotting for intersections and isolated connection segments.

- `get_segment(road_id) -> Segment | None`
  - Returns the segment mapped to a road ID.

- `Segment(lanes, idx=None, concave_hull_ratio=0.3)`
  - Base segment object with polygon generation, caching, lane management, and shared segment plotting.

### OSI Centerline API

- `MapOsiCenterlineSegmentation(recording, lane_buffer=0.3, intersection_overlap_buffer=1.0, concave_hull_ratio=0.3)`
- `init_intersections()`
  - Runs the full OSI centerline segmentation pipeline.
- Inherits shared `plot()` and `plot_intersections()` behavior from `MapSegmentation`
- `SegmentOsiCenterline`
- `Intersection`
- `ConnectionSegment`

### OpenDRIVE API

- `MapODRSegmentation(recording, concave_hull_ratio=0.3)`
- `identify_segments()`
  - Runs the full OpenDRIVE segmentation pipeline.
- Inherits shared `plot()` and `plot_intersections()` behavior from `MapSegmentation`
- `SegmentOdr`
- `IntersectionOdr`
- `ConnectionSegmentOdr`

`Recording.create_mapsegments()` selects `MapOsiCenterlineSegmentation` for `MapOsiCenterline` maps and `MapODRSegmentation` for `MapOdr` maps.


## Plotting

Plotting is implemented in the shared `MapSegmentation` and `Segment` base classes. OSI and ODR subclasses only customize map-specific labels, titles, filenames, and legend entries.

All plot methods support `output_plot=None` to show the plot interactively. `output_plot` may also be:
- an existing directory
- a suffixless directory path that will be created
- a file path ending in `.pdf`, `.png`, or `.svg`

Shared map plotting supports:
- lane centerline coloring
- optional lane ID labels
- optional intersection polygons
- optional connection polygons
- optional trajectory overlay
- traffic-light markers when traffic-light data is available

Shared segment plotting supports lane centerlines, lane labels, and segment polygons for both intersections and connection segments.

Map-specific plotting hooks currently provide:
- OSI map filename/title: `Map_with_Intersection.pdf` / `Map with Intersections`
- ODR map filename/title: `MapODR_Segmentation.pdf` / `MapODR - Segmentation`
- ODR lane labels formatted as `road_id/lane_id`

By default, intersection lanes are green, approaching lanes are orange, and road lanes are black. ODR segmentation does not currently mark approaching lanes, so ODR plots normally show intersection lanes and road lanes only.


## Assumptions and Limitations

- Coordinates are meters in a consistent local frame.
- Lane centerlines should be valid shapely `LineString` geometries.
- Concave hulls at ratio `0.3` should be reasonable for most segments, but irregular geometries may need tuning.
- OSI topology depends on consistent successor/predecessor IDs.
- ODR segmentation depends on OpenDRIVE road `junction` attributes.
- OSI traffic-light assignment is nearest-centerline based.
- ODR traffic-light mapping is currently not implemented.
- `update_road_ids()` mutates OSI lane IDs and map lane dictionary keys; external code relying on original OSI road IDs should account for this.
- Trajectory segmentation is strict: unknown, unassigned, or off-lane points raise errors instead of silently assigning an approximate segment.
- Large maps may require threshold tuning for OSI spatial queries.


## Edge Cases to Consider

- Empty maps produce empty structures; plotting may produce a blank map.
- Degenerate or extremely short centerlines can make polygon generation unstable.
- Dense OSI lane networks may need adjusted `lane_buffer` or `intersection_overlap_buffer`.
- Miswired OSI predecessors/successors can affect approaching-lane marking and connection grouping.
- ODR road and junction IDs may overlap; segment indices remain unique because they are assigned sequentially.
- Trajectory points on lane polygon boundaries are accepted because `covers()` is used.
- Trajectory points outside the located lane polygon raise `RuntimeError`.


## Extensibility Notes

To support a new map type:
1. Create a concrete `Segment` subclass implementing:
   - `_get_lane_id(lane)`
   - `_get_lane_geometry(lane)`
   - `set_trafficlight()`
2. Create a concrete `MapSegmentation` subclass implementing:
   - `_get_lane_id(lane)`
   - `_get_lane_centerline(lane)`
   - `_get_lane_successors(lane)`
   - `_get_lane_predecessors(lane)`
   - `_has_traffic_light(lane)`
   - `_get_traffic_light(lane)`
   - `_set_lane_on_intersection(lane, value)`
   - `_set_lane_is_approaching(lane, value)`
   - `_get_lane_on_intersection(lane)`
3. Override `_located_lane_id_to_segment_lane_id()` if the `Locator` lane ID differs from the `lane_segment_dict` key.
4. Override plotting hooks only when labels, titles, filenames, or legends need map-specific formatting.
5. Add a map-specific segmentation entry point, such as `init_intersections()` or `identify_segments()`.

Potential extensions:
- Add classifiers for ramps, roundabouts, or other `MapSegmentType` values.
- Override polygon generation for map types that need buffered unions or different hull settings.
- Improve traffic-light mapping with explicit stop-line or signal topology.
- Add richer plotting hooks or interactive inspection on top of the shared plot methods.
