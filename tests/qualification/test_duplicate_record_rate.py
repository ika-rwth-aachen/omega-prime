"""."""

import pytest

from omega_prime.qualification.duplicate_record_rate import duplicate_record_rate


def test_pass(duplicate_df_pass) -> None:
    _df, result_dict = duplicate_record_rate(duplicate_df_pass, threshold=1.0)
    summary = result_dict["duplicate_record_rate"].collect()
    assert summary["total_records"][0] == 4
    assert summary["duplicate_records"][0] == 0
    assert summary["duplicate_record_rate"][0] == pytest.approx(0.0)
    assert summary["status"][0] == "pass"


def test_fail(duplicate_df_fail) -> None:
    _df, result_dict = duplicate_record_rate(duplicate_df_fail, threshold=1.0)
    summary = result_dict["duplicate_record_rate"].collect()
    duplicates = result_dict["duplicate_record_duplicates"].collect()
    assert summary["total_records"][0] == 6
    assert summary["duplicate_records"][0] == 4
    assert summary["duplicate_record_rate"][0] == pytest.approx(66.66666666666667)
    assert summary["status"][0] == "fail"
    assert duplicates["duplicate_records"].sum() == 4
