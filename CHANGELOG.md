# 0.3.8

- `Recording` reads, stores and writes the OSI `EnvironmentalConditions` of a recording (MCAP and Parquet), see `Recording.environmental_conditions`, `Recording.set_environmental_conditions` and `Recording.environmental_conditions_df`.
- `Recording.interpolate` returns the recording and keeps traffic light states, environmental conditions and projections.

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
