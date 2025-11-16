"""
Pipeline to merge Vivino wine data with regional and yearly weather datasets.

Steps
-----
1. Load Vivino wine and region CSVs.
2. Extract dominant grape variety from the 'cepages' column.
3. Expand regions across years (2010–2024).
4. Attach nearest weather station data within distance threshold.
5. Merge Vivino with weather data by (region, year).
6. Save merged outputs to CSV and Parquet.

Usage
-----
python -m src.preprocessing.merge_wine_weather
"""

import ast
import logging
from pathlib import Path

import pandas as pd

from src.preprocessing.utils.weather_utils import attach_weather_pandas
from src.utils.config_logger import setup_logging
from src.visualization.plots.plot_metrics import plot_histogram

logger = logging.getLogger(__name__)


def extract_dominant_cepage(value: str | float) -> str | None:
    """Return the dominant grape ('seo_name') from a Vivino 'cepages' literal string."""
    if pd.isna(value):
        return None
    try:
        grape_list = ast.literal_eval(str(value))
        if isinstance(grape_list, list) and grape_list:
            return grape_list[0].get("seo_name")
    except (ValueError, SyntaxError):
        pass
    return None


def merge_wine_and_weather(
    regions_csv: Path,
    vivino_csv: Path,
    weather_dir: Path,
    output_dir: Path,
    max_km: float = 40.0,
    year_range: tuple[int, int] = (2010, 2025),
) -> pd.DataFrame:
    """
    Merge Vivino wines with weather data and save results.

    Parameters
    ----------
    regions_csv : Path
        CSV path with 'region', 'latitude', 'longitude' columns.
    vivino_csv : Path
        CSV path to Vivino dataset.
    weather_dir : Path
        Directory containing weather_cleaned_YYYY.parquet files.
    output_dir : Path
        Directory to save merged outputs.
    max_km : float, optional
        Maximum matching radius in kilometers.
    year_range : tuple[int, int], optional
        Year range (inclusive start, exclusive end).
    """
    logger.info("=== Starting Vivino × Weather merge ===")

    # --- Load inputs ---
    regions = pd.read_csv(regions_csv)
    vivino = pd.read_csv(vivino_csv)

    vivino["cepages"] = vivino["cepages"].apply(extract_dominant_cepage)
    vivino = vivino.dropna(subset=["cepages"])
    vivino["year"] = pd.to_numeric(vivino["vintage"], errors="coerce")

    # --- Expand regions by year ---
    start, end = year_range
    years = pd.DataFrame({"year": range(start, end)})
    regions_expanded = regions.merge(years, how="cross")

    # --- Attach weather ---
    logger.info("Attaching weather data (max radius = %.1f km)...", max_km)
    regions_weather = attach_weather_pandas(regions_expanded, weather_dir, max_km=max_km)

    # --- Merge with Vivino ---
    merged = vivino.merge(regions_weather, on=["region", "year"], how="left")
    merged = merged.dropna(subset=["cepages", "year", "hot_days"])
    logger.info("Merged dataset shape: %s", merged.shape)

    # --- Save outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "vivino_wines_with_weather.csv"
    out_parquet = output_dir / "vivino_wines_with_weather.parquet"

    merged.to_csv(out_csv, index=False)
    merged.to_parquet(out_parquet, index=False)
    logger.info("Saved: %s and %s", out_csv.name, out_parquet.name)

    # --- Optional visualization ---
    if "dist_km" in merged.columns:
        plot_histogram(
            merged["dist_km"],
            bins=50,
            title="Distance to Closest Weather Station",
            xlabel="Distance (km)",
            ylabel="Frequency",
            out_path=output_dir / "weather_distance_hist.html",
            xlim=(0, 40),
        )

    logger.info("=== Merge complete ===")
    return merged


def main():
    setup_logging()
    base = Path("data")
    merge_wine_and_weather(
        regions_csv=base / "wine/regions.csv",
        vivino_csv=base / "Wine/vivino_wines.csv",
        weather_dir=base / "weather_by_year_cleaned",
        output_dir=base / "out",
    )


if __name__ == "__main__":
    main()
