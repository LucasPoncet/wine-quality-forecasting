"""
Weather station matching utilities for region-year dataframes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)
R_EARTH_KM = 6371.0


def attach_weather(
    df: pl.DataFrame, weather_dir: Path, out_csv: Path, max_km: float = 40.0
) -> None:
    """Attach nearest weather station coordinates by year."""
    if weather_dir is None or "year" not in df.columns:
        df.write_csv(out_csv)
        return

    results = []
    for year in sorted({int(y) for y in df["year"].unique() if y is not None}):
        wx_fp = weather_dir / f"weather_cleaned_{year}.parquet"
        if not wx_fp.exists():
            continue

        wx = pl.read_parquet(wx_fp)
        if {"lat", "lon"}.issubset(wx.columns):
            wx = wx.rename({"lat": "latitude", "lon": "longitude"})

        wx = wx.with_columns(
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64),
        )

        tree = BallTree(
            np.deg2rad(wx.select(["latitude", "longitude"]).to_numpy()), metric="haversine"
        )

        subset = df.filter(pl.col("year") == year)
        if subset.is_empty():
            continue

        q = np.deg2rad(subset.select(["latitude", "longitude"]).to_numpy())
        dist, ind = tree.query(q, k=1)

        subset = subset.with_columns(
            pl.Series("wx_station", wx["station"].to_numpy()[ind.flatten()]),
            pl.Series("wx_dist_km", dist.flatten() * 6371.0),
        ).filter(pl.col("wx_dist_km") <= max_km)

        results.append(subset)

    (pl.concat(results) if results else df).write_csv(out_csv)


def attach_weather_pandas(
    regions: pd.DataFrame,
    weather_dir: Path,
    max_km: float = 40.0,
) -> pd.DataFrame:
    """
    Attach nearest weather station data to each (region, year) record.

    Parameters
    ----------
    regions : pd.DataFrame
        Must contain ['region', 'year', 'latitude', 'longitude'].
    weather_dir : Path
        Directory containing weather_cleaned_YYYY.parquet files.
    max_km : float
        Maximum matching radius in kilometers.

    Returns
    -------
    pd.DataFrame
        Regions joined with nearest weather station data.
    """
    out_frames = []
    years = sorted(regions["year"].dropna().unique().astype(int))

    for year in years:
        wx_fp = weather_dir / f"weather_cleaned_{year}.parquet"
        if not wx_fp.exists():
            logger.warning("Missing weather file for year %d", year)
            continue

        wx = pd.read_parquet(wx_fp).dropna(subset=["latitude", "longitude"])
        if wx.empty:
            continue

        tree = BallTree(np.radians(wx[["latitude", "longitude"]].values), metric="haversine")
        subset = regions.query("year == @year and latitude.notna() and longitude.notna()")
        if subset.empty:
            continue

        query_pts = np.radians(subset[["latitude", "longitude"]].values)
        dist_rad, idx = tree.query(query_pts, k=1)
        dist_km = dist_rad.flatten() * R_EARTH_KM

        valid = dist_km <= max_km
        met_cols = [c for c in wx.columns if c not in ("latitude", "longitude", "year")]
        met_block = pd.DataFrame(np.nan, index=subset.index, columns=met_cols)
        met_block.loc[valid, :] = wx.iloc[idx[valid, 0]][met_cols].values

        merged = subset.join(met_block).assign(dist_km=dist_km)
        out_frames.append(merged)

    result = pd.concat(out_frames, ignore_index=True) if out_frames else regions.copy()
    logger.info("Weather data attached: %d records", len(result))
    return result


def attach_weather_polars(
    df: pl.DataFrame,
    weather_dir: Path,
    max_km: float = 40.0,
) -> pl.DataFrame:
    """
    Polars equivalent of attach_weather_pandas.
    """
    out = []
    for year in sorted(set(df["year"].drop_nulls().unique().to_list())):
        wx_fp = weather_dir / f"weather_cleaned_{year}.parquet"
        if not wx_fp.exists():
            continue

        wx = pl.read_parquet(wx_fp)
        if {"lat", "lon"}.issubset(wx.columns):
            wx = wx.rename({"lat": "latitude", "lon": "longitude"})
        wx = wx.drop_nulls(["latitude", "longitude"])

        tree = BallTree(
            np.radians(wx.select(["latitude", "longitude"]).to_numpy()), metric="haversine"
        )
        sub = df.filter(pl.col("year") == year)
        if sub.is_empty():
            continue

        q = np.radians(sub.select(["latitude", "longitude"]).to_numpy())
        dist, idx = tree.query(q, k=1)
        dist_km = dist.flatten() * R_EARTH_KM

        met_cols = [c for c in wx.columns if c not in ("latitude", "longitude", "year")]
        met_block = pl.DataFrame({c: wx[c].to_numpy()[idx.flatten()] for c in met_cols})

        merged = sub.hstack(met_block)
        merged = merged.with_columns(pl.Series("dist_km", dist_km))
        merged = merged.filter(pl.col("dist_km") <= max_km)
        out.append(merged)

    result = pl.concat(out) if out else df
    logger.info("Polars weather join complete: %d rows", result.height)
    return result
