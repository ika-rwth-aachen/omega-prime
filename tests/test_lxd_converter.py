from unittest.mock import MagicMock
import betterosi
import polars as pl
import pytest

from omega_prime.converters.lxd import LxdConverter


def test_lxd_converter_to_omega_prime_recording():
    converter = LxdConverter.__new__(LxdConverter)

    mock_recording = MagicMock()
    mock_recording.get_meta_data.side_effect = lambda key: 25.0 if key == "frameRate" else None
    mock_recording.opendrive_map_file = None

    meta_df = pl.DataFrame(
        {
            "trackId": [1, 2, 3],
            "class": ["bicycle", "pedestrian", "car"],
        }
    )
    tracks_df = pl.DataFrame(
        {
            "trackId": [1, 2, 3],
            "frame": [0, 0, 0],
            "xCenter": [10.0, 20.0, 30.0],
            "yCenter": [15.0, 25.0, 35.0],
            "xVelocity": [1.0, 0.5, 5.0],
            "yVelocity": [0.0, 0.0, 0.0],
            "xAcceleration": [0.1, 0.0, 0.2],
            "yAcceleration": [0.0, 0.0, 0.0],
            "heading": [0.0, 0.0, 0.0],
            "width": [1.0, 0.6, 1.8],
            "length": [1.8, 0.6, 4.5],
        }
    )

    mock_recording._tracks_meta_data = meta_df
    mock_recording._get_tracks_data.return_value = tracks_df

    with pytest.warns(UserWarning, match="No map associated"):
        op_rec = converter.to_omega_prime_recording(mock_recording)

    df = op_rec.df
    assert df.height == 3

    # Check bicycle row
    bicycle_row = df.filter(pl.col("idx") == 1).to_dicts()[0]
    assert bicycle_row["type"] == int(betterosi.MovingObject.Type.VEHICLE)
    assert bicycle_row["subtype"] == int(betterosi.MovingObject.VehicleClassification.Type.BICYCLE)
    assert bicycle_row["width"] == pytest.approx(0.8)
    assert bicycle_row["length"] == pytest.approx(2.0)
    assert bicycle_row["height"] == pytest.approx(1.9)

    # Check pedestrian row
    ped_row = df.filter(pl.col("idx") == 2).to_dicts()[0]
    assert ped_row["type"] == int(betterosi.MovingObject.Type.PEDESTRIAN)
    assert ped_row["subtype"] == -1
    assert ped_row["width"] == pytest.approx(0.5)
    assert ped_row["length"] == pytest.approx(0.5)
    assert ped_row["height"] == pytest.approx(1.8)
