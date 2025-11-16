import geopandas as gpd
import polars as pl
from shapely.geometry import Point

from src.preprocessing.utils.geo_utils import load_aoc


def test_load_aoc_fuzzy(monkeypatch, tmp_path):
    # mock shapefile
    gdf = gpd.GeoDataFrame(
        {
            "denom": ["bordeaux", "champagne", "unknown"],
            "geometry": [Point(1, 44), Point(3, 49), Point(0, 0)],
        },
        crs="EPSG:4326",
    )
    shp_fp = tmp_path / "mock.shp"
    gdf.to_file(shp_fp)

    # mock Vivino
    csv_fp = tmp_path / "vivino.csv"
    pl.DataFrame({"Wine": ["Bordeaux Supérieur", "Champagne Blanc"]}).write_csv(csv_fp)

    out = load_aoc(shp_fp, csv_fp, threshold=70)
    assert "lat" in out.columns
    assert "lon" in out.columns
    assert out.height == 2  # 2 matched entries
