"""
build_wines_coord.py — orchestrator for building wine coordinates pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

import polars as pl

from src.preprocessing.utils.coords_utils import (
    attach_coords,
    compute_barycentres,
    inject_centroids,
)
from src.preprocessing.utils.geo_utils import load_aoc
from src.preprocessing.utils.text_utils import clean_wine_name, normalize_string
from src.preprocessing.utils.weather_utils import attach_weather
from src.utils.config_logger import setup_logging
from src.visualization.maps.plot_wine_map import plot_wine_map

logger = logging.getLogger(__name__)


def load_vivino_dataset(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    return df.with_columns(
        pl.col("Wine").map_elements(clean_wine_name, return_dtype=pl.Utf8).alias("wine_clean"),
        pl.col("region").map_elements(normalize_string, return_dtype=pl.Utf8).alias("region_clean"),
    )


def recap(df: pl.DataFrame) -> None:
    n_total = df.height
    n_with_coords = df.filter(pl.col("latitude").is_not_null()).height
    logger.info("Summary: %d wines, %d with coordinates", n_total, n_with_coords)


def parse_args() -> argparse.Namespace:
    return argparse.Namespace(
        input_csv=Path("data/vivino_wines_with_weather.csv"),
        aoc_shp=Path(
            "data/2025-07-01-delim-parcellaire-aoc-shp/2025-07-01_delim-parcellaire-aoc-shp.shp"
        ),
        weather_dir=Path("data/weather_by_year_cleaned"),
        output_dir=Path("data/out"),
    )


def main():
    setup_logging()
    start = perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corr_fp = Path("data/Wine/region_centroids_from_wines_corrected.csv")

    logger.info("=== Starting build_wines_coord pipeline ===")
    aoc = load_aoc(args.aoc_shp, args.input_csv)
    viv = load_vivino_dataset(args.input_csv)

    viv = attach_coords(viv, aoc).with_columns(
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    )

    bary = compute_barycentres(viv)
    viv = inject_centroids(viv, bary, corr_fp)

    viv.write_csv(args.output_dir / "vivino_with_coords_injected.csv")
    bary.write_csv(args.output_dir / "region_centroids.csv")

    plot_wine_map(viv, args.output_dir / "wine_map.html")
    attach_weather(viv, args.weather_dir, args.output_dir / "vivino_wines_with_weather_AOC.csv")

    recap(viv)
    logger.info("Pipeline completed in %.2f s", perf_counter() - start)


if __name__ == "__main__":
    main()
