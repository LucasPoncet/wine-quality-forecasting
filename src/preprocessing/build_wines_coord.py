"""
pipeline_wine.py  — version corrigée

• Respecte strictement la logique du notebook
  – fuzzy-matching des AOC, reprojection EPSG:2154 → centroïdes WGS84
  – dé-duplication des AOC (un seul point par dénomination)
  – jointure Vivino sans duplication
  – injection des barycentres AVANT la carte
  – compte final ≈ 10 055 vins

Exécution simple :
python pipeline_wine.py
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
from rapidfuzz import fuzz
from sklearn.neighbors import BallTree

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def normalize_string(txt: str) -> str:
    if txt is None:
        return ""
    txt = txt.lower().strip().replace("-", " ")
    txt = re.sub(r"[\"'’`()\[\],.;:!?]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def clean_wine_name(name: str) -> str:
    if name is None:
        return ""
    name = re.sub(r"\b(19|20)\d{2}\b", "", name.lower())
    return normalize_string(name)


# 1. AOC : fuzzy-matching + centroïdes + DÉ-DUP


def load_aoc(shp_path: Path, wine_csv: Path) -> pl.DataFrame:
    gdf = gpd.read_file(shp_path)

    denom_col = (
        "denom"
        if "denom" in gdf.columns
        else ("denominatio" if "denominatio" in gdf.columns else gdf.columns[0])
    )

    gdf["denom_clean"] = (
        gdf[denom_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace("-", " ")
        .str.replace("’", "'")
    )

    viv = pl.read_csv(wine_csv)
    region_list: list[str] = viv.select("Wine").drop_nulls().unique().to_series().to_list()

    def match(denom: str, regions: list[str], thr: int = 85) -> bool:
        return any(fuzz.partial_ratio(denom, r) > thr for r in regions)

    gdf = gdf[gdf["denom_clean"].apply(lambda x: match(x, region_list))].copy()

    gdf_metric = gdf.to_crs(epsg=2154)
    gdf_metric["centroid"] = gdf_metric.geometry.centroid
    gdf_cent = gdf_metric.set_geometry("centroid").set_crs(epsg=2154).to_crs(epsg=4326)

    gdf["lat"] = gdf_cent.geometry.y
    gdf["lon"] = gdf_cent.geometry.x

    aoc = (
        pl.from_pandas(gdf[["denom_clean", "lat", "lon"]])
        .group_by("denom_clean")
        .agg(
            pl.col("lat").mean().alias("lat"),
            pl.col("lon").mean().alias("lon"),
        )
    )

    logger.info("%d AOC filtrées & dédupliquées", aoc.height)
    return aoc


# 2. Vivino cleaning


def load_vivino(csv: Path) -> pl.DataFrame:
    df = pl.read_csv(csv)
    df = df.with_columns(
        pl.col("Wine").map_elements(clean_wine_name, return_dtype=pl.Utf8).alias("wine_clean"),
        pl.col("region").map_elements(normalize_string, return_dtype=pl.Utf8).alias("region_clean"),
    )
    return df


# 3. Attach coordinates from AOC to Vivino wines


def attach_coords(viv: pl.DataFrame, aoc: pl.DataFrame) -> pl.DataFrame:
    viv = viv.join(aoc, left_on="region_clean", right_on="denom_clean", how="left").rename(
        {"lat": "lat_aoc", "lon": "lon_aoc"}
    )

    viv = viv.with_columns(
        pl.when(pl.col("latitude").is_null())
        .then(pl.col("lat_aoc"))
        .otherwise(pl.col("latitude"))
        .alias("latitude"),
        pl.when(pl.col("longitude").is_null())
        .then(pl.col("lon_aoc"))
        .otherwise(pl.col("longitude"))
        .alias("longitude"),
    )
    return viv


# 4. Centroids


def centroids(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.drop_nulls(["latitude", "longitude", "region_clean"])
        .group_by("region_clean")
        .agg(
            pl.col("latitude").mean().alias("cent_lat"),
            pl.col("longitude").mean().alias("cent_lon"),
        )
    )


def inject_centroids(viv: pl.DataFrame, bary: pl.DataFrame, corr_fp: Path) -> pl.DataFrame:
    """
    Priorité des coordonnées, par région :
        1) coordonnées corrigées (fichier externe)
        2) barycentre calculé
        3) coordonnées déjà présentes
    """
    viv = viv.with_columns(
        pl.col("region")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all("-", " ")
        .alias("region_clean")
    )

    p_corr = pd.read_csv(corr_fp)

    # force minuscules / strip
    p_corr["region_clean"] = (
        p_corr["region_clean"].astype(str).str.strip().str.lower().str.replace("-", " ")
    )

    def num(x):
        return re.sub(r"[^\d\.\-]+", "", str(x))

    p_corr["lat_corr"] = pd.to_numeric(p_corr["lat_mean"].apply(num), errors="coerce")
    p_corr["lon_corr"] = pd.to_numeric(p_corr["lon_mean"].apply(num), errors="coerce")

    corr = pl.from_pandas(p_corr[["region_clean", "lat_corr", "lon_corr"]]).drop_nulls(
        ["lat_corr", "lon_corr"]
    )

    viv_upd = viv.join(corr, on="region_clean", how="left")

    viv_upd = viv_upd.with_columns(
        [
            pl.when(pl.col("lat_corr").is_not_null())
            .then(pl.col("lat_corr"))
            .otherwise(pl.col("latitude"))
            .alias("latitude"),
            pl.when(pl.col("lon_corr").is_not_null())
            .then(pl.col("lon_corr"))
            .otherwise(pl.col("longitude"))
            .alias("longitude"),
        ]
    ).drop(["lat_corr", "lon_corr", "region_clean"])

    return viv_upd


# 5. Plotly map


def make_map(df: pl.DataFrame, out_html: Path):
    pdf = df.drop_nulls(["latitude", "longitude"]).to_pandas()
    if pdf.empty:
        logger.warning("Carte non générée (aucun point)")
        return
    fig = px.scatter_map(
        pdf,
        lat="latitude",
        lon="longitude",
        hover_name="Wine",
        zoom=4,
        height=700,
        title="Carte des vins (coords finales)",
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(out_html)
    logger.info("Carte sauvegardée : %s", out_html)


# 6. Weather


def attach_weather(df: pl.DataFrame, weather_dir: Path | None, out_csv: Path, max_km: float = 40.0):
    if weather_dir is None or "year" not in df.columns:
        df.write_csv(out_csv)
        return

    out = []
    for year in sorted({int(y) for y in df["year"].unique() if y is not None}):
        wx_fp = weather_dir / f"weather_cleaned_{year}.parquet"
        if not wx_fp.exists():
            continue
        wx = pl.read_parquet(wx_fp)
        if {"lat", "lon"}.issubset(wx.columns):
            wx = wx.rename({"lat": "latitude", "lon": "longitude"})
        wx = wx.with_columns(
            pl.col("latitude").cast(pl.Float64), pl.col("longitude").cast(pl.Float64)
        )
        tree = BallTree(
            np.deg2rad(wx.select(["latitude", "longitude"]).to_numpy()),
            metric="haversine",
        )

        sub = df.filter(pl.col("year") == year)
        if sub.is_empty():
            continue
        q = np.deg2rad(sub.select(["latitude", "longitude"]).to_numpy())
        dist, ind = tree.query(q, k=1)
        sub = sub.with_columns(
            pl.Series("wx_station", wx["station"].to_numpy()[ind.flatten()]),
            pl.Series("wx_dist_km", dist.flatten() * 6371.0),
        ).filter(pl.col("wx_dist_km") <= max_km)
        out.append(sub)

    (pl.concat(out) if out else df).write_csv(out_csv)


def recap(df: pl.DataFrame):
    n = df.height
    n_coord = df.filter(pl.col("latitude").is_not_null()).height
    logger.info("Récapitulatif : %d vins, %d avec coordonnées", n, n_coord)


def parse_args():
    return argparse.Namespace(
        input_csv=Path("data/vivino_wines_with_weather.csv"),
        aoc_shp=Path(
            "data/2025-07-01-delim-parcellaire-aoc-shp/2025-07-01_delim-parcellaire-aoc-shp.shp"
        ),
        weather_dir=Path("data/weather_by_year_cleaned"),
        output_dir=Path("data/out"),
    )


def main():
    start = perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    CORR_CENT_FP = Path("data/Wine/region_centroids_from_wines_corrected.csv")

    aoc = load_aoc(args.aoc_shp, args.input_csv)
    viv = load_vivino(args.input_csv)
    viv = attach_coords(viv, aoc)
    viv = viv.with_columns(
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    )

    bary = centroids(viv)
    viv = inject_centroids(viv, bary, CORR_CENT_FP)

    viv.write_csv(args.output_dir / "vivino_with_coords_injected.csv")
    bary.write_csv(args.output_dir / "region_centroids.csv")

    make_map(viv, args.output_dir / "wine_map.html")

    attach_weather(viv, args.weather_dir, args.output_dir / "vivino_wines_with_weather_AOC.csv")

    recap(viv)
    elapsed = perf_counter() - start
    logger.info("Pipeline terminé en %.2f s", elapsed)


if __name__ == "__main__":
    main()
