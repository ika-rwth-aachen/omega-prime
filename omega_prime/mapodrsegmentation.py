import logging
from collections import defaultdict, namedtuple as nt
from pathlib import Path

import numpy as np
import shapely
from matplotlib import pyplot as plt

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


class IntersectionOdr(SegmentOdr):
    """Represents an OpenDRIVE junction (intersection)."""

    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_junction_id=None):
        super().__init__(lanes, idx, concave_hull_ratio=concave_hull_ratio)
        self.odr_junction_id = odr_junction_id
        self.type = MapSegmentType.JUNCTION

    def plot(self, output_plot: Path = None):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.set_title(f"Intersection {self.idx}")
        for lane in self.lanes:
            ax.plot(*np.asarray(lane.centerline.xy)[:2], color="green")
        for lane in self.lanes:
            m = int(np.ceil(len(lane.centerline.xy[0]) / 2))
            ax.annotate(
                f"{lane.idx.road_id}/{lane.idx.lane_id}",
                xy=(lane.centerline.xy[0][m], lane.centerline.xy[1][m]),
                fontsize=2,
                color="black",
                zorder=3,
            )
        try:
            ax.fill(*self.polygon.exterior.xy, color="green", alpha=0.2, zorder=5)
            ax.plot(*self.polygon.exterior.xy, color="green", alpha=0.7, zorder=10)
        except Exception:
            logger.warning(f"IntersectionOdr {self.idx} has no polygon")
        plt.title(f"Intersection {self.idx} ({len(self.lanes)} lanes)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        _save_or_show(output_plot, f"IntersectionOdr{self.idx}.pdf")


class ConnectionSegmentOdr(SegmentOdr):
    """Represents a non-junction road segment in an OpenDRIVE map."""

    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3, odr_road_id=None):
        super().__init__(lanes, idx, concave_hull_ratio=concave_hull_ratio)
        self.odr_road_id = odr_road_id
        self.type = MapSegmentType.STRAIGHT

    def plot(self, output_plot: Path = None):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.set_title(f"Connection {self.idx}")
        for lane in self.lanes:
            ax.plot(*np.asarray(lane.centerline.xy)[:2], color="steelblue")
        for lane in self.lanes:
            m = int(np.ceil(len(lane.centerline.xy[0]) / 2))
            ax.annotate(
                f"{lane.idx.road_id}/{lane.idx.lane_id}",
                xy=(lane.centerline.xy[0][m], lane.centerline.xy[1][m]),
                fontsize=2,
                color="black",
                zorder=3,
            )
        try:
            ax.fill(*self.polygon.exterior.xy, color="steelblue", alpha=0.2, zorder=5)
            ax.plot(*self.polygon.exterior.xy, color="steelblue", alpha=0.7, zorder=10)
        except Exception:
            logger.warning(f"ConnectionSegmentOdr {self.idx} has no polygon")
        plt.title(f"Connection {self.idx} ({len(self.lanes)} lanes)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        _save_or_show(output_plot, f"ConnectionSegmentOdr{self.idx}.pdf")


def _save_or_show(output_plot, filename: str):
    """Save figure to *output_plot* directory/file, or show it if *output_plot* is None."""
    if output_plot is None:
        plt.show()
    else:
        output_path = Path(output_plot)
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / filename)
        elif output_path.suffix in (".pdf", ".png", ".svg"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
        else:
            raise ValueError(f"output_plot must be a directory or a file path (.pdf/.png/.svg), got: {output_plot}")
    plt.close()


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
                logger.warning(f"Could not create IntersectionOdr for junction {junction_id}: {e}")

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
                logger.warning(f"Could not create ConnectionSegmentOdr for road {road_id}: {e}")

        # Step 5: store segments with IDs from one namespace. OpenDRIVE road and
        # junction IDs can overlap, so keep those source IDs separately.
        self.intersections = intersections
        self.isolated_connections = isolated_connections
        self.segments = intersections + isolated_connections
        for idx, segment in enumerate(self.segments):
            segment.idx = idx

        self.build_segment_by_road_id()

        self.create_lane_segment_dict()
        self.check_if_all_lanes_are_on_segment()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        output_plot: Path = None,
        trajectory=None,
        plot_lane_ids: bool = False,
        plot_intersection_polygons: bool = True,
        plot_connection_polygons: bool = False,
    ):
        """Plot the segmented map with colour-coded lanes and segment outlines.

        Args:
            output_plot: Directory or file path to save the figure. If ``None``
                the figure is shown interactively.
            trajectory: Optional ``np.ndarray`` of shape ``(n, 3)`` with columns
                ``(frame, x, y)`` to overlay on the map.
            plot_lane_ids: Annotate each lane with its ``road_id/lane_id``.
            plot_intersection_polygons: Draw the convex hull of each
                intersection in red.
            plot_connection_polygons: Draw the convex hull of each connection
                segment in blue.
        """
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)

        # Lane centerlines – colour by intersection / approaching status
        for lane in self.lanes.values():
            if lane.on_intersection:
                c = "green"
            elif getattr(lane, "is_approaching", None):
                c = "orange"
            else:
                c = "black"
            ax.plot(*lane.centerline.xy, color=c, alpha=0.4, zorder=-10)

        if plot_lane_ids:
            for lane in self.lanes.values():
                mid = lane.centerline.interpolate(0.5, normalized=True)
                ax.annotate(
                    f"{lane.idx.road_id}/{lane.idx.lane_id}",
                    xy=(mid.x, mid.y),
                    fontsize=2,
                    color="black",
                )

        # Intersections
        for inter in self.intersections:
            ax.annotate(inter.idx, xy=inter.get_center_point(), fontsize=4, color="darkgreen", fontweight="bold")
            if plot_intersection_polygons:
                inter.update_polygon()
                ax.fill(*inter.polygon.exterior.xy, color="green", alpha=0.15, zorder=5)
                ax.plot(*inter.polygon.exterior.xy, color="green", alpha=0.7, zorder=10, linewidth=1)

        # Connection segments
        for conn in self.isolated_connections:
            ax.annotate(conn.idx, xy=conn.get_center_point(), fontsize=4, color="steelblue")
            if plot_connection_polygons:
                conn.update_polygon()
                try:
                    ax.fill(*conn.polygon.exterior.xy, color="steelblue", alpha=0.1, zorder=5)
                    ax.plot(*conn.polygon.exterior.xy, color="steelblue", alpha=0.5, zorder=10, linewidth=0.8)
                except Exception:
                    logger.warning(f"ConnectionSegmentOdr {conn.idx} has no plottable polygon")

        # Trajectory overlay
        if trajectory is not None:
            ax.plot(trajectory[:, 1], trajectory[:, 2], color="yellow", alpha=0.8, linewidth=2, label="Trajectory")
            ax.plot(trajectory[0, 1], trajectory[0, 2], "go", markersize=8, label="Start")
            ax.plot(trajectory[-1, 1], trajectory[-1, 2], "ro", markersize=8, label="End")

        plt.title("MapODR – Segmentation")
        plt.xlabel("X Coordinate (m)", fontsize=10)
        plt.ylabel("Y Coordinate (m)", fontsize=10)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color="green", label="Intersection lane"),
                plt.Line2D([0], [0], color="orange", label="Approaching lane"),
                plt.Line2D([0], [0], color="black", label="Road lane"),
            ],
            fontsize=7,
        )
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        _save_or_show(output_plot, "MapODR_Segmentation.pdf")

    def plot_intersections(self, output_plot: Path = None):
        """Save one plot per intersection and connection segment.

        Args:
            output_plot: Directory to save individual plots into, or ``None``
                to show each interactively.
        """
        for intersection in self.intersections:
            intersection.plot(output_plot)
        for connection in self.isolated_connections:
            connection.plot(output_plot)
