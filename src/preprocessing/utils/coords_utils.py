import logging
import re
from pathlib import Path

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def attach_coords(viv: pl.DataFrame, aoc: pl.DataFrame) -> pl.DataFrame:
    """Attach AOC coordinates to Vivino wines."""
    viv = viv.join(aoc, left_on="region_clean", right_on="denom_clean", how="left")
    viv = viv.rename({"lat": "lat_aoc", "lon": "lon_aoc"})

    return viv.with_columns(
        pl.when(pl.col("latitude").is_null())
        .then(pl.col("lat_aoc"))
        .otherwise(pl.col("latitude"))
        .alias("latitude"),
        pl.when(pl.col("longitude").is_null())
        .then(pl.col("lon_aoc"))
        .otherwise(pl.col("longitude"))
        .alias("longitude"),
    ).drop(["lat_aoc", "lon_aoc"])


def compute_barycentres(df: pl.DataFrame) -> pl.DataFrame:
    """Compute average lat/lon per region."""
    return (
        df.drop_nulls(["latitude", "longitude", "region_clean"])
        .group_by("region_clean")
        .agg(
            pl.col("latitude").mean().alias("cent_lat"),
            pl.col("longitude").mean().alias("cent_lon"),
        )
    )


def inject_centroids(viv: pl.DataFrame, bary: pl.DataFrame, corr_fp: Path) -> pl.DataFrame:
    """Inject corrected or barycentric coordinates."""
    viv = viv.with_columns(
        pl.col("region")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all("-", " ")
        .alias("region_clean")
    )

    corr_df = pd.read_csv(corr_fp)
    corr_df["region_clean"] = (
        corr_df["region_clean"].astype(str).str.strip().str.lower().str.replace("-", " ")
    )

    def num(x):
        return re.sub(r"[^\d\.\-]+", "", str(x))

    corr_df["lat_corr"] = pd.to_numeric(corr_df["lat_mean"].apply(num), errors="coerce")
    corr_df["lon_corr"] = pd.to_numeric(corr_df["lon_mean"].apply(num), errors="coerce")

    corr = pl.from_pandas(corr_df[["region_clean", "lat_corr", "lon_corr"]]).drop_nulls()

    viv_upd = viv.join(corr, on="region_clean", how="left")

    return viv_upd.with_columns(
        pl.when(pl.col("lat_corr").is_not_null())
        .then(pl.col("lat_corr"))
        .otherwise(pl.col("latitude"))
        .alias("latitude"),
        pl.when(pl.col("lon_corr").is_not_null())
        .then(pl.col("lon_corr"))
        .otherwise(pl.col("longitude"))
        .alias("longitude"),
    ).drop(["lat_corr", "lon_corr", "region_clean"])
