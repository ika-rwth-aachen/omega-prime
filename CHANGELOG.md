# Unreleased

- `ShapelyTrajectoryTools.trajectory_axis(x, y, yaw, l_append, still)`: the reference axis of a trajectory, its path
  extended at both ends. A trajectory standing throughout (`still`) or with a path shorter than `min_length` gets an
  axis through its first valid position along its first valid yaw; None without path and yaw.
  `curvilinear_projection` uses it.
- Breaking: `extend_linestring(heading=..., min_length=...)` extends a point, an empty line or a path shorter than
  `min_length` (0.1 m) along `heading` and raises `ValueError` without one. Before, a point or an empty line was
  extended by 1 along the y-axis and a short path along its own geometry.
- `curvilinear_projection`: an ego without path and yaw gives NaN instead of an axis in a made-up direction.
- `Locator`: a lane is extended along its own geometry however short; a lane without a centerline direction (a point,
  an empty line or coinciding points) is skipped with a warning, so there are fewer lanes and the internal lane
  indices shift. Routes through a skipped lane are lost.
- `MapOdr`: lanes without geometry are dropped with a warning and their predecessors linked to their successors.
  pyxodr gives a lane section shorter than `step_size` (at most one sample) zero coordinates, a lane at the origin.
  The `Locator` therefore no longer skips any lane of an ASAM OpenDRIVE map.

# 0.3.7

- Cap `polars<2`: polars-st is not yet compatible with polars 2 (metaclass conflict on import).

# 0.3.6

- Calculate TTC (Time to Collision) and THW (Time Headway) in omega-prime with Polars.

# 0.3.4

 - Add the `qualify` command to CLI.
 - Added metrics to the qualification methodology
   - Attribute completeness
   - Non-default attribute accuracy
   - Object type coverage
   - Record completeness
   - Temporal completeness
   - Class completeness
   - Duplicate record rate
- Refactored metrics to be more modular and reusable
- Added tests for existing metrics, metric-class and metrics-manager
