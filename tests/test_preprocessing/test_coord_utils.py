import pandas as pd
import polars as pl

from src.preprocessing.utils.coords_utils import (
    attach_coords,
    compute_barycentres,
    inject_centroids,
)


def test_attach_coords_merges_successfully():
    viv = pl.DataFrame(
        {
            "region_clean": ["a", "b"],
            "latitude": [None, 40.0],
            "longitude": [None, 3.0],
        }
    )
    aoc = pl.DataFrame(
        {
            "denom_clean": ["a", "b"],
            "lat": [45.0, 50.0],
            "lon": [1.0, 2.0],
        }
    )
    out = attach_coords(viv, aoc)
    assert all(c in out.columns for c in ["latitude", "longitude"])
    assert abs(out["latitude"][0] - 45.0) < 1e-6


def test_compute_barycentres_groups_correctly():
    df = pl.DataFrame(
        {
            "region_clean": ["x", "x", "y"],
            "latitude": [1.0, 3.0, 10.0],
            "longitude": [2.0, 4.0, 20.0],
        }
    )
    bary = compute_barycentres(df)
    assert set(bary.columns) == {"region_clean", "cent_lat", "cent_lon"}
    assert abs(bary.filter(pl.col("region_clean") == "x")["cent_lat"][0] - 2.0) < 1e-6


def test_inject_centroids_with_corrections(tmp_path):
    viv = pl.DataFrame({"region": ["bordeaux"], "latitude": [45.0], "longitude": [0.0]})
    bary = pl.DataFrame({"region_clean": ["bordeaux"], "cent_lat": [46.0], "cent_lon": [1.0]})
    corr_fp = tmp_path / "corr.csv"
    pd.DataFrame({"region_clean": ["bordeaux"], "lat_mean": ["47.0"], "lon_mean": ["2.0"]}).to_csv(
        corr_fp, index=False
    )

    out = inject_centroids(viv, bary, corr_fp)
    assert abs(out["latitude"][0] - 47.0) < 1e-6
