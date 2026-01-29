"""."""

import polars as pl
from collections.abc import Sequence
from omega_prime import MovingObjectType

from ..metrics import metric


@metric(computes_properties=["object_type_coverage"])
def object_type_coverage(
    df: pl.LazyFrame | pl.DataFrame,
    /,
    expected_types: Sequence[MovingObjectType],
):
    present_types = df.select("type").unique().collect().to_series()
    et_set = set(expected_types)
    otc = sum(et in present_types for et in et_set) * 100.0 / len(et_set)

    summary = pl.DataFrame(
        {
            "object_type_coverage": otc,
            "status": ["pass" if otc > 99.9 else "fail"],
        }
    ).lazy()

    return df, {"object_type_coverage": summary}
