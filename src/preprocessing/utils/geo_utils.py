import logging
from pathlib import Path

import geopandas as gpd
import polars as pl
from rapidfuzz import fuzz

from src.preprocessing.utils.text_utils import normalize_string

logger = logging.getLogger(__name__)


def load_aoc(shp_path: Path, wine_csv: Path, threshold: int = 85) -> pl.DataFrame:
    """Fuzzy-match AOC shapefile names with Vivino wines, compute centroids."""
    gdf = gpd.read_file(shp_path)
    denom_col = (
        "denom"
        if "denom" in gdf.columns
        else ("denominatio" if "denominatio" in gdf.columns else gdf.columns[0])
    )

    gdf["denom_clean"] = gdf[denom_col].astype(str).map(normalize_string)

    viv = pl.read_csv(wine_csv)
    regions = viv.select("Wine").drop_nulls().unique().to_series().to_list()

    def is_match(denom: str) -> bool:
        return any(fuzz.partial_ratio(denom, r) > threshold for r in regions)

    gdf = gdf[gdf["denom_clean"].apply(is_match)]
    gdf_metric = gdf.to_crs(epsg=2154)
    gdf_metric["centroid"] = gdf_metric.geometry.centroid
    gdf_cent = gdf_metric.set_geometry("centroid").set_crs(epsg=2154).to_crs(epsg=4326)

    gdf["lat"] = gdf_cent.geometry.y
    gdf["lon"] = gdf_cent.geometry.x

    aoc = (
        pl.from_pandas(gdf[["denom_clean", "lat", "lon"]])
        .group_by("denom_clean")
        .agg(pl.col("lat").mean().alias("lat"), pl.col("lon").mean().alias("lon"))
    )

    logger.info("%d AOC filtered & deduplicated", aoc.height)
    return aoc
