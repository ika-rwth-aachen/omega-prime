import logging
import numpy as np
import networkx as nx
import shapely
from shapely.strtree import STRtree
from shapely.geometry import Point
from collections import namedtuple as nt
from omega_prime.locator import Locator
from matplotlib import pyplot as plt
from pathlib import Path
import shapely
import numpy as np
import networkx as nx
import logging

from .mapsegment import MapSegmentType, Segment, MapSegmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def add_lanexy_to_graph(G : nx.Graph, lanes):
    """
    Adds lane coordinates to the graph as node attributes.

    Args:
        G (networkx.Graph): The graph to which lane coordinates will be added.
        lanes (dict): A dictionary of lane objects.

    Returns:
        networkx.Graph: The updated graph with lane coordinates as node attributes.
    """
    for lane in lanes.values():
        if lane.idx.lane_id in G.nodes:
            G.nodes[lane.idx.lane_id]["x"] = shapely.centroid(lane.centerline).x
            G.nodes[lane.idx.lane_id]["y"] = shapely.centroid(lane.centerline).y
    return G


def plot_graph(G : nx.Graph, output : Path):
    """
    Plots the graph with lane coordinates.

    Args:
        G (networkx.Graph): The graph to be plotted.
        Path (str or Path): The file path to save the plot.
    """
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    nx.draw(G, pos, with_labels=True, node_size=10, font_size=5)
    plt.title("Intersection Graph")
    plt.xlabel("X Coordinate")  # Add label for the x-axis
    plt.ylabel("Y Coordinate")  # Add label for the y-axis
    plt.grid(True)  # Optional: Add a grid for better visualization
    plt.savefig(Path)
    plt.close()  # Close the plot to free memory


# Concrete implementations for OSI Centerline maps
class SegmentOsiCenterline(Segment):
    """Segment implementation for OSI centerline-based maps."""

    def _get_lane_id(self, lane):
        return lane.idx.lane_id

    def _get_lane_geometry(self, lane) -> shapely.LineString:
        return lane.centerline

    def set_trafficlight(self):
        trafficlight = []
        for lane in self.lanes:
            if hasattr(lane, 'trafficlight') and lane.trafficlight:
                trafficlight.append(lane.trafficlight)
        self.trafficlights = trafficlight


class MapOsiCenterlineSegmentation(MapSegmentation):
    """
    A class that identifies different segments on a OsiCenterline Map.
    Concrete implementation of MapSegmentation for OSI centerline maps.
    """

    def __init__(self, recording, lane_buffer=None, intersection_overlap_buffer=None, concave_hull_ratio=0.3):
        super().__init__(recording, concave_hull_ratio=concave_hull_ratio)
        self.locator = Locator.from_map(recording.map)
        self.isolated_connections = []
        self.G = None
        self.lane_buffer = lane_buffer if lane_buffer is not None else 0.3
        self.intersection_overlap_buffer = intersection_overlap_buffer if intersection_overlap_buffer is not None else 1
        self.do_combine_intersections = True
        
        for tl_state in recording.traffic_light_states.values():
            for tl in tl_state:
                if tl.id.value not in self.trafficlight_ids:
                    self.trafficlight[tl.id.value] = tl
                    self.trafficlight_ids.add(tl.id.value)

    # Implement abstract methods for OSI centerline maps
    def _get_lane_id(self, lane):
        """Extract lane ID from OSI centerline lane."""
        return lane.idx.lane_id

    def _get_lane_centerline(self, lane) -> shapely.LineString:
        """Extract centerline from OSI centerline lane."""
        return lane.centerline

    def _get_lane_successors(self, lane) -> list:
        """Extract successor IDs from OSI centerline lane."""
        return [succ_id.lane_id if hasattr(succ_id, "lane_id") else succ_id 
                for succ_id in lane.successor_ids]

    def _get_lane_predecessors(self, lane) -> list:
        """Extract predecessor IDs from OSI centerline lane."""
        return [pred_id.lane_id if hasattr(pred_id, "lane_id") else pred_id 
                for pred_id in lane.predecessor_ids]

    def _has_traffic_light(self, lane) -> bool:
        """Check if OSI centerline lane has traffic light."""
        return hasattr(lane, 'trafficlight') and lane.trafficlight is not None

    def _get_traffic_light(self, lane):
        """Get traffic light object from OSI centerline lane."""
        return lane.trafficlight if self._has_traffic_light(lane) else None

    def _set_lane_on_intersection(self, lane, value: bool):
        """Set the on_intersection attribute for OSI centerline lane."""
        lane.on_intersection = value

    def _set_lane_is_approaching(self, lane, value: bool):
        """Set the is_approaching attribute for OSI centerline lane."""
        lane.is_approaching = value

    def _get_lane_on_intersection(self, lane) -> bool:
        """Get the on_intersection status of OSI centerline lane."""
        return lane.on_intersection if hasattr(lane, 'on_intersection') else False

    def init_intersections(self):
        """
        Initializes the intersections in the map.
        Args:
            None
        Returns:
            None
        """
        self.create_lane_dict()
        self.get_lane_successors_and_predecessors()
        self.parallel_lane_dict = self.create_parallel_lane_dict()
        self.get_intersecting_lanes()
        self.set_lane_trafficlights()
        self.graph_intersection_detection()
        self.G = add_lanexy_to_graph(self.G, self.lanes)
        self.set_intersection_idx()

        if self.do_combine_intersections:
            self.add_non_intersecting_lanes_to_intersection()
            self.combine_intersections()
            self.set_intersection_idx()
            self.create_intersection_dict()

        self.create_lane_segment_dict()
        self.find_isolated_connections()
        self.create_lane_segment_dict()
        self.check_if_all_lanes_are_on_segment()
        self.update_segment_ids()
        self.create_lane_segment_dict()
        self.update_road_ids()
        self.set_lane_intersection_relation()

        # from pathlib import Path
        # #Plot the graph G with x and y coordinates of the lanes
        # plot_graph(self.G , Path("/scenario-center-playground/scenarios/") / "graph_plot.pdf")

    def update_road_ids(self):
        """
        Updates the road_ids of the lane to the segment ID
        """
        updates_needed = []
        old_to_new_mapping = {}

        # First pass: identify what needs to be updated
        for lane_idx, lane in self.lanes.items():
            lane_id = lane.idx.lane_id
            if lane_id in self.lane_segment_dict and self.lane_segment_dict[lane_id].segment is not None:
                new_road_id = self.lane_segment_dict[lane_id].segment.idx
                if lane.idx.road_id != new_road_id:
                    new_idx = lane.idx._replace(road_id=new_road_id)
                    updates_needed.append((lane_idx, lane, new_idx))
                    old_to_new_mapping[lane_idx] = new_idx

        # Second pass: apply updates efficiently
        for old_idx, lane, new_idx in updates_needed:
            # Update the lane object in place
            lane.idx = new_idx

            # Only modify dictionary if the key actually changed
            if old_idx != new_idx:
                self.lanes[new_idx] = lane
                del self.lanes[old_idx]

        # Third pass: update all predecessor and successor references
        for lane in self.lanes.values():
            # Update predecessor references
            updated_predecessors = []
            for pred_id in lane.predecessor_ids:
                if pred_id in old_to_new_mapping:
                    updated_predecessors.append(old_to_new_mapping[pred_id])
                else:
                    updated_predecessors.append(pred_id)
            lane.predecessor_ids = updated_predecessors

            # Update successor references
            updated_successors = []
            for succ_id in lane.successor_ids:
                if succ_id in old_to_new_mapping:
                    updated_successors.append(old_to_new_mapping[succ_id])
                else:
                    updated_successors.append(succ_id)
            lane.successor_ids = updated_successors

        # Fourth pass: update internal dictionaries that track relationships
        self.lane_dict = {lane.idx.lane_id: lane for lane in self.lanes.values()}
        self.get_lane_successors_and_predecessors()

    def update_segment_ids(self):
        "Updates the segment IDs of the map segmentation"
        self.segments = self.intersections + self.isolated_connections
        for i, segment in enumerate(self.segments):
            segment.idx = i
            segment.set_trafficlight()

    def create_parallel_lane_dict(self):
        """
        Creates a dictionary mapping each lane's lane_id to the lane ids which are parallel to it
        Args:
            None
        Returns:
            dict: A dictionary mapping each lane's lane_id to the lane ids which are parallel to it.
        """
        lane_dict = {lane.idx.lane_id: [] for lane in self.lanes.values()}

        # Precompute lane directions for faster comparisons
        lane_directions = {}
        lane_centerlines = []
        lane_ids = []

        for lane in self.lanes.values():
            coords = np.array(lane.centerline.coords)
            direction = coords[-1] - coords[0]
            lane_directions[lane.idx.lane_id] = direction / np.linalg.norm(direction)
            lane_centerlines.append(lane.centerline)
            lane_ids.append(lane.idx.lane_id)

        if not lane_centerlines:
            return lane_dict

        # Use original centerlines for spatial index, buffer only when needed
        tree = STRtree(lane_centerlines)

        for i, lane in enumerate(self.lanes.values()):
            lane_id = lane.idx.lane_id

            # Create buffer only when querying, not storing it
            buffer_geom = lane.centerline.buffer(10)
            candidates = tree.query(buffer_geom)

            # Clear the buffer immediately after use
            del buffer_geom

            for idx in candidates:
                other_lane_id = lane_ids[idx]
                if other_lane_id == lane_id:
                    continue

                # Compare directions using dot product
                dir1 = lane_directions[lane_id]
                dir2 = lane_directions[other_lane_id]
                dot_product = np.clip(np.abs(np.dot(dir1, dir2)), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot_product))

                if angle_deg < 10:
                    lane_dict[lane_id].append(other_lane_id)

        return lane_dict

    def trajectory_segment_detection(self, trajectory):
        """
        Splits a trajectory into segments based on the lane it is located on

        Args:
            trajectory (np.ndarray): A NumPy array of shape (n, 3) representing the trajectory, where each row is a (frame, x, y) coordinate.

        Returns:
            list: A list of tuples, where each tuple contains a segment of the trajectory and the segment it intersects with.
        """
        segments = []
        current_segment = []
        xy = trajectory[:, 1:3]  # Extract x and y coordinates
        sts = self.locator.xys2sts(xy)
        lane_ids = sts["roadlane_id"].to_numpy()
        segment_idx = [self.lane_segment_dict[lane_id.lane_id].segment.idx for lane_id in lane_ids]

        trajectory = np.column_stack((trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lane_ids, segment_idx))

        # Create spatial index for intersection polygons
        intersection_polygons = []
        intersection_ids = []
        buffer = 5

        for segment in self.segments:
            if segment.type == MapSegmentType.JUNCTION and hasattr(segment, "polygon"):
                intersection_polygons.append(segment.polygon.buffer(buffer))
                intersection_ids.append(segment.idx)

        if intersection_polygons:
            # Use spatial index for efficient intersection queries
            tree = STRtree(intersection_polygons)

            # Process points in batches for better performance
            for i, (frame, x, y, _, _) in enumerate(trajectory):
                point = Point(x, y)

                # Query spatial index instead of checking all polygons
                candidates = tree.query(point)

                for idx in candidates:
                    if intersection_polygons[idx].contains(point):
                        trajectory[i, 4] = intersection_ids[idx]
                        break

        # Rest of the method for creating segments
        prev_seg_id = -1
        for i, (frame, x, y, _, segment_idx) in enumerate(trajectory):
            if prev_seg_id == segment_idx:
                current_segment.append((frame, x, y))
            else:
                if current_segment:
                    segments.append((np.array(current_segment), self.segments[prev_seg_id]))
                current_segment = [(frame, x, y)]
                prev_seg_id = segment_idx

        if current_segment:
            segments.append((np.array(current_segment), self.segments[prev_seg_id]))

        return segments

    def get_intersecting_lanes(self, buffer: float = None):
        """
        Returns a dictionary mapping each lane's lane_id to an array of lane ids it intersects with.

        Args:
            lanes (list): Array of lane objects, each with an `idx` and `centerline` attribute.

        Returns:
            dict: A dictionary where keys are lane ids and values are arrays of intersecting lane ids.
        """
        if buffer is None:
            buffer = self.lane_buffer

        # Create spatial index directly from centerlines
        lane_centerlines = []
        lane_ids = []

        for lane in self.lanes.values():
            lane_centerlines.append(lane.centerline)
            lane_ids.append(lane.idx.lane_id)

        if not lane_centerlines:
            self.intersecting_lanes_dict = {}
            return {}

        tree = STRtree(lane_centerlines)

        # Pre-compute lane relationships for faster lookup
        successors_set = {lane_id: set(successors) for lane_id, successors in self.lane_successors_dict.items()}
        predecessors_set = {lane_id: set(predecessors) for lane_id, predecessors in self.lane_predecessors_dict.items()}
        parallel_set = {lane_id: set(parallel) for lane_id, parallel in self.parallel_lane_dict.items()}

        intersecting_lanes = {}
        for i, lane in enumerate(self.lanes.values()):
            lane_id = lane.idx.lane_id

            # Query spatial index with buffered geometry
            buffered_centerline = lane.centerline.buffer(buffer)
            candidates = tree.query(buffered_centerline)

            intersecting_lanes[lane_id] = []
            for idx in candidates:
                candidate_id = lane_ids[idx]
                if (
                    candidate_id != lane_id
                    and candidate_id not in successors_set[lane_id]
                    and candidate_id not in predecessors_set[lane_id]
                    and candidate_id not in parallel_set[lane_id]
                    and buffered_centerline.intersects(lane_centerlines[idx])
                ):
                    # Only buffer and test intersection for valid candidates
                    intersecting_lanes[lane_id].append(candidate_id)

        self.intersecting_lanes_dict = intersecting_lanes
        return intersecting_lanes

    def graph_intersection_detection(self):
        """
        Detects intersections in a graph of lanes based on their intersections, successors, and predecessors.

        Args:
            lane_dict (dict): A dictionary where keys are lane indices and values are arrays of intersecting lane indices.
            lane_successors (dict): A dictionary where keys are lane indices and values are arrays of successor lane indices.
            lane_predecessors (dict): A dictionary where keys are lane indices and values are arrays of predecessor lane indices.

        Returns:
            list: A list of intersections, where each intersection is a set of lane indices.
        """
        # Create a Graph using networkx
        G = nx.Graph()

        # Add nodes and edges to the graph. If a lane has a intersection, add the lanes as nodes and the intersection as an edge
        # Add edges directly (nodes are added automatically)
        for lane_id, intersecting_lanes in self.intersecting_lanes_dict.items():
            G.add_edges_from((lane_id, other_lane) for other_lane in intersecting_lanes)

        intersections = []
        for inter in nx.connected_components(G):
            # Convert lane_ids back to lane objects
            intersection_lanes = [self.lane_dict[i] for i in inter]
            intersection = Intersection(intersection_lanes, concave_hull_ratio=self.concave_hull_ratio)
            intersections.append(intersection)

        self.intersections = intersections
        self.G = G
        return intersections, G

    def combine_intersections(self):
        """A function that revieves a list with idx [[1,2] , [4,5,6] , ...] of intersections that need to be combined.
        It will combine all those intersections and will then update all intersections in the map_segmentation class.

        Args:
            intersection_list (list): A list of lists, where each inner list contains the indices of intersections to be combined.

        Returns:
            None
        """

        # Check for intersections that can be combined:
        combined_intersections = []

        # Create spatial index of all intersection polygons
        for intersection in self.intersections:
            if (
                not hasattr(intersection, "_buffered_polygon")
                or intersection._buffer_value != self.intersection_overlap_buffer
            ):
                intersection._buffered_polygon = intersection.polygon.buffer(self.intersection_overlap_buffer)
                intersection._buffer_value = self.intersection_overlap_buffer

        polygons = [intersection._buffered_polygon for intersection in self.intersections]
        if polygons:
            tree = STRtree(polygons)

            # Find overlapping intersections efficiently
            for i, intersection in enumerate(self.intersections):
                buffered_poly = polygons[i]
                candidates = tree.query(buffered_poly)

                for j in candidates:
                    if i != j and buffered_poly.intersects(polygons[j]):
                        combined_intersections.append([i, j])
        final_combined = self.find_resulting_intersections(combined_intersections)
        new_intersections = []
        visited = set()

        for combination in final_combined:
            combined_lanes = []
            for idx in combination:
                if idx not in visited:
                    visited.add(idx)
                    combined_lanes.extend(self.intersections[idx].lanes)

            new_intersections.append(Intersection(combined_lanes, concave_hull_ratio=self.concave_hull_ratio))

        # Add unvisited intersections
        for i, intersection in enumerate(self.intersections):
            if i not in visited:
                new_intersections.append(intersection)

        self.intersections = new_intersections

    def intersections_overlap(self, intersection1, intersection2, buffer: float = None):
        """
        Check if two intersections overlap.

        Args:
            intersection1 (Intersection): The first intersection object.
            intersection2 (Intersection): The second intersection object.

        Returns:
            bool: True if the intersections overlap, False otherwise.
        """
        if buffer is None:
            buffer = self.intersection_overlap_buffer

        # Use cached buffers if available
        if not hasattr(intersection1, "_buffered_polygon") or intersection1._buffer_value != buffer:
            intersection1._buffered_polygon = intersection1.polygon.buffer(buffer)
            intersection1._buffer_value = buffer

        if not hasattr(intersection2, "_buffered_polygon") or intersection2._buffer_value != buffer:
            intersection2._buffered_polygon = intersection2.polygon.buffer(buffer)
            intersection2._buffer_value = buffer

        return intersection1.polygon.buffer(buffer).intersects(intersection2.polygon.buffer(buffer))

    def combine_intersection_on_polygon(self, intersection1, intersection2):
        """
        Combine two intersections into one if they overlap.
        Args:
            intersection1 (Intersection): The first intersection object.
            intersection2 (Intersection): The second intersection object.
        Returns:
            Intersection: The combined intersection object if they overlap, None otherwise.
        """
        if self.intersections_overlap(intersection1, intersection2):
            # Create a new intersection object with the lanes from both intersections
            combined_intersection = Intersection(intersection1.lanes + intersection2.lanes, concave_hull_ratio=self.concave_hull_ratio)

            return combined_intersection
        else:
            return None

    def find_resulting_intersections(self, intersection_pairs):
        G = nx.Graph()
        G.add_edges_from(intersection_pairs)
        return [list(component) for component in nx.connected_components(G)]

    def set_intersection_idx(self):
        """
        Sets the index for each intersection in the list of intersections.
        Args:
            None
        Returns:
            None
        """
        for i, intersection in enumerate(self.intersections + self.isolated_connections):
            intersection.idx = i

    def create_intersection_dict(self):
        """Creats a dictionary where the key is the intersection id and the value is the intersection object.
        Args:
            None
        Returns:
            None
        """
        intersection_dict = {}
        for i, intersection in enumerate(self.intersections):
            intersection_dict[intersection.idx] = intersection
        self.intersection_dict = intersection_dict
        return intersection_dict

    def add_non_intersecting_lanes_to_intersection(self):
        """Add all lanes that are within the intersection polygon to the intersection object.
        Args:
            None
        Returns:
            None
        """
        for intersection in self.intersections:
            intersection.update_polygon()

            # Collect all lanes to add before modifying the intersection
            lanes_to_add = []
            buffered_polygon = intersection.polygon.buffer(self.lane_buffer)

            for lane in self.lanes.values():
                lane_id = lane.idx.lane_id
                if (
                    lane_id not in intersection.lane_ids
                    and self.lane_segment_dict[lane_id].segment is None
                    and buffered_polygon.contains(lane.centerline)
                ):
                    lanes_to_add.append(lane)

            # Add all lanes at once and update polygon only once
            if lanes_to_add:
                intersection.add_lane(lanes=lanes_to_add, update_polygon=True)  # Assuming bulk add method

    def create_lane_segment_dict(self):
        """
        Create a dictionary mapping lane IDs to their segment information.
        Args:
            None
        Returns:
            lane_segment_dict (dict): A dictionary mapping lane IDs to their segment information.
        """
        segment_name = nt("SegmentName", ["lane_id", "segment_idx", "segment"])
        segment_list = self.intersections + self.isolated_connections

        # Initialize with None values more efficiently
        lane_segment_dict = {lane_id: segment_name(lane_id, None, None) for lane_id in self.lane_dict.keys()}

        for segment in segment_list:
            for lane in segment.lanes:
                lane_id = lane.idx.lane_id

                # Single lookup with caching
                current_entry = lane_segment_dict.get(lane_id)
                if current_entry is None:
                    continue

                if current_entry.segment is None:
                    # Lane not assigned to any segment yet
                    lane_segment_dict[lane_id] = segment_name(lane_id, segment.idx, segment)
                elif current_entry.segment_idx != segment.idx:
                    # Conflict: lane already assigned to different segment
                    logger.warning(
                        f"Lane {lane_id} already in segment {current_entry.segment_idx}, "
                        f"cannot assign to segment {segment.idx}"
                    )

        self.lane_segment_dict = lane_segment_dict

    def create_non_intersecting_lane_graph(self):
        """Create a graph with each lane which is not part of a intersection as a node and the edges are the successors and predecessors of the lanes.
        Args:
            None
        Returns:
            G (networkx.Graph): A graph with each lane as a node and the edges are the successors and predecessors of the lanes.
        """
        G = nx.Graph()
        for lane in self.lanes.values():
            lane_id = lane.idx.lane_id
            if lane_id not in self.lane_segment_dict or self.lane_segment_dict[lane_id].segment is None:
                G.add_node(lane_id)
                for successor in self.lane_successors_dict[lane_id]:
                    if successor not in self.lane_segment_dict or self.lane_segment_dict[successor].segment is None:
                        G.add_edge(lane_id, successor)
                for predecessor in self.lane_predecessors_dict[lane_id]:
                    if predecessor not in self.lane_segment_dict or self.lane_segment_dict[predecessor].segment is None:
                        G.add_edge(lane_id, predecessor)
        return G

    def find_isolated_connections(self):
        """Find all isolated strings of connections in the graph. Then Check if any of those lanes would be part of an intersection.
        Args:
            None
        Returns:
            isolated_connections (list): A list of lists, where each inner list contains the indices of lanes that are part of an isolated connection.
        """
        G = self.create_non_intersecting_lane_graph()
        isolated_connections = []
        new_connections = []
        for component in nx.connected_components(G):
            if len(component) > 0:
                isolated_connections.append(ConnectionSegment([self.lane_dict[i] for i in component], concave_hull_ratio=self.concave_hull_ratio))
        # Check if any of the lanes in the isolated connections are part of an intersection
        for connection in isolated_connections:
            pre = False
            suc = False
            for lane_id in connection.lane_ids:
                # Check if the lane has a predecessor or successor that is part of an intersection
                for successor in self.lane_successors_dict[lane_id]:
                    if successor in self.lane_segment_dict and self.lane_segment_dict[successor].segment is not None:
                        connection.intersection_idxs.add(self.lane_segment_dict[successor].segment_idx)
                        suc = True
                for predecessor in self.lane_predecessors_dict[lane_id]:
                    if (
                        predecessor in self.lane_segment_dict
                        and self.lane_segment_dict[predecessor].segment is not None
                    ):
                        connection.intersection_idxs.add(self.lane_segment_dict[predecessor].segment_idx)
                        pre = True

            if len(connection.intersection_idxs) == 1 and pre and suc:
                # There is a predecessor and a successor that are part of an intersection so the connection is part of the intersection:
                # Add all the lanes to the intersection:
                for lane_id in connection.lane_ids:
                    self.intersection_dict[list(connection.intersection_idxs)[0]].lanes.append(self.lane_dict[lane_id])
                    self.intersection_dict[list(connection.intersection_idxs)[0]].lane_ids.append(lane_id)
                    self.intersection_dict[list(connection.intersection_idxs)[0]].update_polygon()
            else:
                new_connections.append(connection)

        isolated_connections = new_connections
        # Create ConnectionSegment for all lanes, that are on multiple intersections:

        isolated_connections = self.combine_isolated_connections(isolated_connections)
        return isolated_connections

    def combine_isolated_connections(self, isolated_connections):
        """Check if any of the isolated connections are connecting the same intersections.
        If yes, then combine them into one connection.
        Args:
            isolated_connections (list): A list of ConnectionSegment objects representing isolated connections.
        Returns:
            isolated_connections (list): A list of ConnectionSegment objects representing the combined isolated connections.
        """
        if not isolated_connections:
            return []

        # Group connections by their intersection indices for efficient comparison
        connections_by_intersections = {}
        for i, connection in enumerate(isolated_connections):
            key = frozenset(connection.intersection_idxs)
            if key not in connections_by_intersections:
                connections_by_intersections[key] = []
            connections_by_intersections[key].append(i)

        combined_connections = []

        # Process each group of connections with same intersection indices
        for intersection_set, connection_indices in connections_by_intersections.items():
            if len(connection_indices) > 1:
                if len(intersection_set) > 1:
                    # Multiple intersections: combine all connections
                    for i in range(len(connection_indices)):
                        for j in range(i + 1, len(connection_indices)):
                            combined_connections.append([connection_indices[i], connection_indices[j]])
                else:
                    # Single intersection: check distance
                    connections_to_check = [isolated_connections[idx] for idx in connection_indices]
                    for i in range(len(connections_to_check)):
                        for j in range(i + 1, len(connections_to_check)):
                            if connections_to_check[i].polygon.distance(connections_to_check[j].polygon) < 5:
                                combined_connections.append([connection_indices[i], connection_indices[j]])

        # Rest of the method remains the same
        final_combined = self.find_resulting_intersections(combined_connections)
        new_connections = []
        visited = set()

        for combination in final_combined:
            combined_lanes = []
            for idx in combination:
                if idx not in visited:
                    visited.add(idx)
                    combined_lanes.extend(isolated_connections[idx].lanes)
            new_connections.append(ConnectionSegment(combined_lanes, concave_hull_ratio=self.concave_hull_ratio))

        # Add unvisited connections
        for i, connection in enumerate(isolated_connections):
            if i not in visited:
                new_connections.append(connection)

        self.isolated_connections = new_connections
        return self.isolated_connections

    def set_lane_intersection_relation(self):
        """
        Sets the attribute lane.is_approaching true if the lane is connecting to an intersection.
        Sets the attribute lane.on_intersection true if the lane is part of an intersection.
        """
        for lane in self.lanes.values():
            self._set_lane_on_intersection(lane, False)
            self._set_lane_is_approaching(lane, False)

        # Process intersection lanes and their predecessors efficiently
        for intersection in self.intersections:
            # Mark intersection lanes
            for lane in intersection.lanes:
                lane_id = self._get_lane_id(lane)
                if lane_id in self.lanes:
                    self._set_lane_on_intersection(self.lanes[lane_id], True)

                # Process predecessors for each lane in the intersection
                for predecessor_id in self._get_lane_predecessors(lane):
                    if predecessor_id in self.lane_dict:
                        predecessor = self.lane_dict[predecessor_id]
                        if not self._get_lane_on_intersection(predecessor):
                            self._set_lane_is_approaching(predecessor, True)

    def set_lane_trafficlights(self):
        """
        Sets the traffic lights for each lane of the map.
        """
        # Create spatial index for lane centerlines
        lane_centerlines = [lane.centerline for lane in self.lanes.values()]
        lane_objects = list(self.lanes.values())

        tree = STRtree(lane_centerlines)

        for tl_idx in self.trafficlight:
            traffic_light_found = False

            # Create traffic light position point
            tl_point = Point(self.trafficlight[tl_idx].base.position.x, self.trafficlight[tl_idx].base.position.y)

            # Use spatial index to find candidate lanes
            candidates = tree.nearest(tl_point)

            if candidates:
                lane = lane_objects[candidates]
                lane.traffic_light = self.trafficlight[tl_idx]
                traffic_light_found = True

            if not traffic_light_found:
                logger.warning(f"Traffic light {self.trafficlight[tl_idx].id} not found in any lane")

    def plot(
        self,
        output_plot: Path = None,
        trajectory=None,
        plot_lane_ids=False,
        plot_intersection_polygons=False,
        plot_connection_polygons=False,
    ):
        """
        Plots the intersections and saves the plot to the specified output path.
        A Trajectory can be given to plot it on the map. The Trajectory should be a numpy array of shape (n,3) where each row is (frame, x, y)
        Args:
            output_plot (Path): Path to a folder where the plot will be saved. If None, the plot will be shown instead.
            trajectory (numpy.ndarray): The trajectory to be plotted. If None, no trajectory will be plotted.
            plot_lane_ids (bool): Whether to plot lane IDs on the map.
            plot_intersection_polygons (bool): Whether to plot intersection polygons.
            plot_connection_polygons (bool): Whether to plot connection polygons.
        Returns:
            None
        """
        # Plot the map by plotting all the centerlines:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)

        for lane in self.lanes.values():
            c = "blue"
            if lane.on_intersection:
                c = "green"
            elif lane.is_approaching:
                c = "orange"
            else:
                c = "black"
            ax.plot(*lane.centerline.xy, color=c, alpha=0.3, zorder=-10)

        if plot_lane_ids:
            lane_midpoints = [
                (lane.idx, lane.centerline.interpolate(0.5, normalized=True)) for lane in self.lanes.values()
            ]
            for lane_id, midpoint in lane_midpoints:
                ax.annotate(lane_id, xy=(midpoint.x, midpoint.y), fontsize=2, color="black")

        for inter in self.intersections:
            ax.annotate(inter.idx, xy=inter.get_center_point(), fontsize=2, color="black")

            if plot_intersection_polygons:
                # Plot the polygon into the intersection
                inter.update_polygon()
                ax.plot(*inter.polygon.exterior.xy, color="red", alpha=0.5, zorder=10)

        for combi in self.isolated_connections:
            ax.annotate(combi.idx, xy=combi.get_center_point(), fontsize=2, color="black")
            # Plot the polygon into the intersection
            if plot_connection_polygons:
                combi.update_polygon()
                try:
                    ax.plot(*combi.polygon.exterior.xy, color="blue", alpha=0.5, zorder=10)
                except:
                    logger.warning(f"Connection {combi.idx} has no polygon")
                    pass

        for tl_idx in self.trafficlight:
            position = shapely.Point(
                self.trafficlight[tl_idx].base.position.x, self.trafficlight[tl_idx].base.position.y
            )
            ax.plot(
                position.x,
                position.y,
                marker="o",
                color="red",
                markersize=2,
                label=f"Traffic Light {self.trafficlight[tl_idx].id}",
            )

        # Plot the trajectory if it is given
        if trajectory is not None:
            plt.plot(
                trajectory[:, 1],
                trajectory[:, 2],
                color="yellow",
                alpha=0.8,
                linewidth=3,
                label="Host Vehicle Trajectory",
            )

            # Mark start and end points
            plt.plot(trajectory[0, 1], trajectory[0, 2], "go", markersize=10, label="Start")
            plt.plot(trajectory[-1, 1], trajectory[-1, 2], "ro", markersize=10, label="End")

        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())
        plt.title("Map with Intersections")
        plt.xlabel("X Coordinate (m)", fontsize=12)
        plt.ylabel("Y Coordinate (m)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        if output_plot is None:
            plt.show()
        else:
            if isinstance(output_plot, Path):
                output_plot.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_plot / "Map_with_Intersection.pdf")
            else:
                isinstance(output_plot, str)
                output_path = Path(output_plot)
                if output_path.is_dir():
                    output_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path / "Map_with_Intersection.pdf")
                elif output_path.suffix == ".pdf":
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path)
        plt.close()

    def plot_intersections(self, output_plot : Path):
        """
        Plots all intersections and saves them to the output path.
        Args:
            output_plot (Path): Path to a folder where the plots will be saved.
        Returns:
            None
        """
        for i, intersection in enumerate(self.intersections):
            intersection.plot(output_plot)
        for i, connection in enumerate(self.isolated_connections):
            connection.plot(output_plot)


class Intersection(SegmentOsiCenterline):
    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3):
        super().__init__(lanes, idx, concave_hull_ratio=concave_hull_ratio)
        self.type = MapSegmentType.JUNCTION

    def plot(self, output_plot: Path):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        # Add the index of the center line to the plot
        ax.set_title(f"Intersection {self.idx}")
        for lane in self.lanes:
            ax.plot(*np.asarray(lane.centerline.xy)[:2], color="blue")
        for lane in self.lanes:
            m = int(np.ceil(len(lane.centerline.xy[0]) / 2))
            ax.annotate(
                lane.idx.lane_id,
                xy=(lane.centerline.xy[0][m], lane.centerline.xy[1][m]),
                fontsize=2,
                color="black",
                zorder=3,
            )
        # Plot the polygon into the intersection
        try:
            ax.plot(*self.polygon.exterior.xy, color="red", alpha=0.5, zorder=10)
        except:
            logging.warning(f"Intersection {self.idx} has no polygon")
            pass
        ax.set_aspect(1)
        plt.title(f"Intersection with {len(self.lanes)} lanes")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        if output_plot is None:
            plt.show()
        elif isinstance(output_plot, Path) and output_plot.is_dir():
            output_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_plot / f"Intersection{self.idx}.pdf")
        else:
            raise ValueError("output_plot must be a Path to a directory or None")
        plt.close()


class ConnectionSegment(SegmentOsiCenterline):
    def __init__(self, lanes, idx=None, concave_hull_ratio=0.3):
        super().__init__(lanes, idx, concave_hull_ratio=concave_hull_ratio)
        self.type = MapSegmentType.STRAIGHT
        self.intersection_idxs = set()

    def plot(self, output_plot: Path):
        """Plots the Connection segment

        Args:
            output_plot (Path): Path to the output directory.
        Returns:
            None
        """
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        # Add the index of the center line to the plot
        ax.set_title(f"Connection segment {self.idx}")
        for lane in self.lanes:
            ax.plot(*np.asarray(lane.centerline.xy)[:2], color="blue")
        for lane in self.lanes:
            m = int(np.ceil(len(lane.centerline.xy[0]) / 2))
            ax.annotate(
                lane.idx.lane_id,
                xy=(lane.centerline.xy[0][m], lane.centerline.xy[1][m]),
                fontsize=2,
                color="black",
                zorder=3,
            )
        # Plot the polygon into the intersection
        try:
            ax.plot(*self.polygon.exterior.xy, color="red", alpha=0.5, zorder=10)
        except:
            logging.warning(f"Connection {self.idx} has no polygon")
            pass
        ax.set_aspect(1)
        plt.title(f"Connection with {len(self.lanes)} lanes")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        if output_plot is None:
            plt.show()
        elif isinstance(output_plot, Path) and output_plot.is_dir():
            output_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_plot / f"Connection{self.idx}.pdf")
        else:
            raise ValueError("output_plot must be a Path to a directory or None")
        plt.close()
