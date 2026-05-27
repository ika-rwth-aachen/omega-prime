import logging
from collections import defaultdict, namedtuple as nt
import shapely
from omega_prime.map_odr import MapOdr
from omega_prime.mapsegment import MapSegmentation, MapSegmentType, Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ODR-specific Segment classes
# ---------------------------------------------------------------------------


class SegmentOdr(Segment):
    """Concrete Segment base for OpenDRIVE-based maps.

    Uses the full XodrLaneId (road_id, lane_id, section_id) as the lane key
    to guarantee uniqueness across roads (bare integer lane IDs repeat in every
    road of an ODR file).
    """

    def _get_lane_id(self, lane):
        return lane.idx  # full XodrLaneId namedtuple

    def _get_lane_geometry(self, lane) -> shapely.LineString:
        return lane.centerline

    def set_trafficlight(self):
        # MapOdr carries no traffic-light data currently – no-op stub
        pass

    def _plot_lane_label(self, lane):
        return f"{lane.idx.road_id}/{lane.idx.lane_id}"


class IntersectionOdr(SegmentOdr):
    """Represents an OpenDRIVE junction (intersection)."""

    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_junction_id=None):
        super().__init__(lanes, idx, concave_hull_ratio=concave_hull_ratio)
        self.odr_junction_id = odr_junction_id
        self.type = MapSegmentType.JUNCTION



class ConnectionSegmentOdr(SegmentOdr):
    """Represents a non-junction road segment in an OpenDRIVE map."""

    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_road_id=None):
        super().__init__(lanes, idx, concave_hull_ratio=concave_hull_ratio)
        self.odr_road_id = odr_road_id
        self.type = MapSegmentType.STRAIGHT




# ---------------------------------------------------------------------------
# MapODRSegmentation
# ---------------------------------------------------------------------------


class MapODRSegmentation(MapSegmentation):
    """Identifies intersections and road segments in an OpenDRIVE (MapOdr) map.

    Uses the ``junction`` attribute already present on every OpenDRIVE road
    element to group lanes: roads whose ``junction`` attribute differs from
    ``"-1"`` belong to that junction and are classified as intersection lanes.
    No graph-based detection is required.
    """

    def __init__(self, recording, concave_hull_ratio=0.3):
        assert recording.map is not None, (
            "Recording does not contain a map. Please provide a recording with a map to use MapODRSegmentation."
        )
        assert isinstance(recording.map, MapOdr), (
            "Map in recording is not of type MapOdr. Please provide a recording "
            "with a MapOdr map to use MapODRSegmentation."
        )
        super().__init__(recording, concave_hull_ratio=concave_hull_ratio)

    # ------------------------------------------------------------------
    # Abstract method implementations (MapSegmentation interface)
    # ------------------------------------------------------------------

    def _get_lane_id(self, lane):
        return lane.idx  # full XodrLaneId

    def _get_lane_centerline(self, lane) -> shapely.LineString:
        return lane.centerline

    def _get_lane_successors(self, lane) -> list:
        return lane.successor_ids

    def _get_lane_predecessors(self, lane) -> list:
        return lane.predecessor_ids

    def _has_traffic_light(self, lane) -> bool:
        return False

    def _get_traffic_light(self, lane):
        return None

    def _set_lane_on_intersection(self, lane, value: bool):
        lane.on_intersection = value

    def _set_lane_is_approaching(self, lane, value: bool):
        lane.is_approaching = value

    def _get_lane_on_intersection(self, lane) -> bool:
        return lane.on_intersection

    def _plot_lane_label(self, lane):
        return f"{lane.idx.road_id}/{lane.idx.lane_id}"

    def _plot_map_title(self):
        return "MapODR - Segmentation"

    def _plot_map_filename(self):
        return "MapODR_Segmentation.pdf"

    def _plot_legend_handles(self):
        from matplotlib import pyplot as plt

        return [
            plt.Line2D([0], [0], color="green", label="Intersection lane"),
            plt.Line2D([0], [0], color="black", label="Road lane"),
        ]

    # ------------------------------------------------------------------
    # ODR-specific helpers
    # ------------------------------------------------------------------

    def _build_road_junction_mapping(self) -> dict:
        """Return a mapping ``road_id → junction_id`` for every junction road.

        Only roads where ``road.road_xml.get("junction") != "-1"`` are included.
        """
        mapping = {}
        for road in self.map.xodr_map.get_roads():
            junction_id = road.road_xml.get("junction")
            if junction_id is not None and junction_id != "-1":
                mapping[road.id] = junction_id
        return mapping

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def create_lane_segment_dict(self):
        """Map every lane's ``XodrLaneId`` to its segment.

        Overrides the base-class version to key on the full ``XodrLaneId``
        (rather than a bare integer lane_id).
        """
        segment_name = nt("SegmentName", ["lane_id", "segment_idx", "segment"])
        segment_list = self.intersections + self.isolated_connections

        lane_segment_dict = {lane_id: segment_name(lane_id, None, None) for lane_id in self.lane_dict.keys()}

        for segment in segment_list:
            for lane in segment.lanes:
                lane_id = lane.idx
                current = lane_segment_dict.get(lane_id)
                if current is None:
                    continue
                if current.segment is None:
                    lane_segment_dict[lane_id] = segment_name(lane_id, segment.idx, segment)
                elif current.segment_idx != segment.idx:
                    logger.warning(
                        f"Lane {lane_id} already in segment {current.segment_idx}, "
                        f"cannot assign to segment {segment.idx}"
                    )

        self.lane_segment_dict = lane_segment_dict
        return lane_segment_dict

    def identify_segments(self):
        """Main entry point: detect intersections and connection segments.

        Steps
        -----
        1. Build helper dicts (lane dict, successor/predecessor dicts).
        2. Read the OpenDRIVE ``junction`` attribute to map road IDs to
           junction IDs.
        3. Group ``on_intersection`` lanes by junction ID → one
           ``IntersectionOdr`` per junction.
        4. Group remaining lanes by ``road_id`` → one ``ConnectionSegmentOdr``
           per road.
        5. Assign unique sequential segment indices and build ``lane_segment_dict``.
        """
        self.create_lane_dict()
        self.get_lane_successors_and_predecessors()

        # Step 2: road_id → junction_id (only for junction roads)
        road_junction_map = self._build_road_junction_mapping()

        # Step 3: group intersection lanes by junction_id
        junction_lanes: dict[str, list] = defaultdict(list)
        for lane in self.lanes.values():
            road_id = lane.idx.road_id
            if road_id in road_junction_map:
                junction_id = road_junction_map[road_id]
                junction_lanes[junction_id].append(lane)

        segment_failures = []
        intersections = []
        for junction_id, lanes in junction_lanes.items():
            try:
                seg = IntersectionOdr(
                    lanes,
                    concave_hull_ratio=self.concave_hull_ratio,
                    odr_junction_id=junction_id,
                )
                intersections.append(seg)
            except Exception as e:
                message = f"Could not create IntersectionOdr for ODR junction {junction_id}: {e}"
                logger.warning(message)
                segment_failures.append(message)

        # Step 4: group non-junction lanes by road_id → ConnectionSegmentOdr
        junction_road_ids = set(road_junction_map.keys())
        road_lanes: dict[str, list] = defaultdict(list)
        for lane in self.lanes.values():
            road_id = lane.idx.road_id
            if road_id not in junction_road_ids:
                road_lanes[road_id].append(lane)

        isolated_connections = []
        for road_id, lanes in road_lanes.items():
            try:
                seg = ConnectionSegmentOdr(
                    lanes,
                    concave_hull_ratio=self.concave_hull_ratio,
                    odr_road_id=road_id,
                )
                isolated_connections.append(seg)
            except Exception as e:
                message = f"Could not create ConnectionSegmentOdr for ODR road {road_id}: {e}"
                logger.warning(message)
                segment_failures.append(message)

        # Step 5: store segments with IDs from one namespace. OpenDRIVE road and
        # junction IDs can overlap, so keep those source IDs separately.
        self.intersections = intersections
        self.isolated_connections = isolated_connections
        self.segments = intersections + isolated_connections
        for idx, segment in enumerate(self.segments):
            segment.idx = idx

        self.build_segment_by_road_id()

        self.create_lane_segment_dict()
        unassigned_lane_ids = [lane_id for lane_id, entry in self.lane_segment_dict.items() if entry.segment is None]
        if segment_failures or unassigned_lane_ids:
            details = []
            if segment_failures:
                details.append("segment creation failures: " + "; ".join(segment_failures))
            if unassigned_lane_ids:
                preview = ", ".join(str(lane_id) for lane_id in unassigned_lane_ids[:20])
                if len(unassigned_lane_ids) > 20:
                    preview += f", ... ({len(unassigned_lane_ids)} total)"
                details.append("unassigned lanes: " + preview)
            raise RuntimeError("MapODRSegmentation did not assign all lanes to segments: " + " | ".join(details))

