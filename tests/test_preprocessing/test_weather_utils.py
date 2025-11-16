import polars as pl

from src.preprocessing.utils.weather_utils import attach_weather


def test_attach_weather_simple(tmp_path):
    # mock weather parquet
    weather_dir = tmp_path / "wx"
    weather_dir.mkdir()
    wx = pl.DataFrame(
        {
            "station": ["A1", "A2"],
            "lat": [45.0, 47.0],
            "lon": [1.0, 3.0],
        }
    )
    wx.write_parquet(weather_dir / "weather_cleaned_2020.parquet")

    df = pl.DataFrame(
        {
            "year": [2020],
            "latitude": [45.1],
            "longitude": [1.1],
        }
    )

    out_csv = tmp_path / "out.csv"
    attach_weather(df, weather_dir, out_csv, max_km=500)
    out = pl.read_csv(out_csv)
    assert "wx_station" in out.columns
    assert out["wx_dist_km"][0] < 500
