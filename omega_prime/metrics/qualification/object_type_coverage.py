"""."""

from collections.abc import Sequence

import polars as pl

from ..metric import metric
from ...types import MovingObjectType
from .common import STATUS, PASS, FAIL, QRT

OBJECT_TYPE_COVERAGE = "object_type_coverage"


@metric(computes_properties=[OBJECT_TYPE_COVERAGE])
def object_type_coverage(df: pl.LazyFrame, /, expected_types: Sequence[MovingObjectType]) -> QRT:
    present_types = df.select("type").unique().collect().to_series()
    et_set = set(expected_types)
    otc = sum(et in present_types for et in et_set) * 100.0 / len(et_set)

    summary = pl.DataFrame(
        {
            OBJECT_TYPE_COVERAGE: otc,
            STATUS: [PASS if otc > 99.9 else FAIL],
        }
    ).lazy()

    return df, {OBJECT_TYPE_COVERAGE: summary}
