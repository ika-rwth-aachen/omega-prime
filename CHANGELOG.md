# 0.3.9

Long recordings (a continuous drive of several minutes on a long ASAM OpenDRIVE road) no longer run out of time or
memory. Results without the new parameters are unchanged (for equal predicted timegaps the reported object
timestamp `p_timegaps.total_nanos` depends on row order, as it did before).

- `Locator`: building it and locating points take time linear in the vertices of a lane, not quadratic. The vertex
  distances come from the segment lengths, `needs_angle_adjustment` from the first vertex (it decides the result for
  all of them), and lanes are tested as prepared geometries after a bounding-box query of the tree.
- `timegaps`: only the columns it uses go into the cross join with the time window (which polars runs as a
  range join); it carried every column of both sides before. The expression is unchanged, so are the results.
- `p_timegaps`: new parameter `overlap_horizon` [ns]. With it only overlaps within the horizon of the ego timestamp
  are used, computed block by block; without it (default) every overlap of the pair is used, as before.
- `Recording.from_file(step_size=...)` also reaches an ASAM OpenDRIVE map embedded in an MCAP file.
- Note: `needs_angle_adjustment` holds for nearly every point beyond the first vertex of the line (it compares
  against the smallest vertex distance). Kept as it was; it may have been meant as `|lpd - s| <= epsi`.

# 0.3.8

- `Recording` reads, stores and writes the OSI `EnvironmentalConditions` of a recording (MCAP and Parquet), see `Recording.environmental_conditions`, `Recording.set_environmental_conditions` and `Recording.environmental_conditions_df`.
- `Recording.interpolate` returns the recording and keeps traffic light states, environmental conditions and projections.
- `Recording.interpolate` takes type, subtype and role from the nearest sample in time linear in the samples of a track; the distance matrix of all pairs did not fit into memory for a long track.

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
