# Importation
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

dfr = pd.read_csv("data/wine/regions.csv")
dfr["latitude"] = pd.to_numeric(dfr["latitude"], errors="coerce")
dfr["longitude"] = pd.to_numeric(dfr["longitude"], errors="coerce")

dfv = pd.read_csv("data/Wine/vivino_wines.csv")

DATA_WEATHER = Path("data/weather_by_year_cleaned")
df10 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2010.parquet")
df11 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2011.parquet")
df12 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2012.parquet")
df13 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2013.parquet")
df14 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2014.parquet")
df15 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2015.parquet")
df16 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2016.parquet")
df17 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2017.parquet")
df18 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2018.parquet")
df19 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2019.parquet")
df20 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2020.parquet")
df21 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2021.parquet")
df22 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2022.parquet")
df23 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2023.parquet")
df24 = pd.read_parquet(f"{DATA_WEATHER}/weather_cleaned_2024.parquet")

df10["latitude"] = pd.to_numeric(df10["latitude"], errors="coerce")
df10["longitude"] = pd.to_numeric(df10["longitude"], errors="coerce")
df11["latitude"] = pd.to_numeric(df11["latitude"], errors="coerce")
df11["longitude"] = pd.to_numeric(df11["longitude"], errors="coerce")
df12["latitude"] = pd.to_numeric(df12["latitude"], errors="coerce")
df12["longitude"] = pd.to_numeric(df12["longitude"], errors="coerce")
df13["latitude"] = pd.to_numeric(df13["latitude"], errors="coerce")
df13["longitude"] = pd.to_numeric(df13["longitude"], errors="coerce")
df14["latitude"] = pd.to_numeric(df14["latitude"], errors="coerce")
df14["longitude"] = pd.to_numeric(df14["longitude"], errors="coerce")
df15["latitude"] = pd.to_numeric(df15["latitude"], errors="coerce")
df15["longitude"] = pd.to_numeric(df15["longitude"], errors="coerce")
df16["latitude"] = pd.to_numeric(df16["latitude"], errors="coerce")
df16["longitude"] = pd.to_numeric(df16["longitude"], errors="coerce")
df17["latitude"] = pd.to_numeric(df17["latitude"], errors="coerce")
df17["longitude"] = pd.to_numeric(df17["longitude"], errors="coerce")
df18["latitude"] = pd.to_numeric(df18["latitude"], errors="coerce")
df18["longitude"] = pd.to_numeric(df18["longitude"], errors="coerce")
df19["latitude"] = pd.to_numeric(df19["latitude"], errors="coerce")
df19["longitude"] = pd.to_numeric(df19["longitude"], errors="coerce")
df20["latitude"] = pd.to_numeric(df20["latitude"], errors="coerce")
df20["longitude"] = pd.to_numeric(df20["longitude"], errors="coerce")
df21["latitude"] = pd.to_numeric(df21["latitude"], errors="coerce")
df21["longitude"] = pd.to_numeric(df21["longitude"], errors="coerce")
df22["latitude"] = pd.to_numeric(df22["latitude"], errors="coerce")
df22["longitude"] = pd.to_numeric(df22["longitude"], errors="coerce")
df23["latitude"] = pd.to_numeric(df23["latitude"], errors="coerce")
df23["longitude"] = pd.to_numeric(df23["longitude"], errors="coerce")
df24["latitude"] = pd.to_numeric(df24["latitude"], errors="coerce")
df24["longitude"] = pd.to_numeric(df24["longitude"], errors="coerce")


# Main cepage selection


def extract_dominant(sep_str: str | float) -> str | None:
    """Return first-grape 'seo_name' from a cepages literal‐string.

    Handles NaNs, empty lists, or malformed rows safely.
    """
    if pd.isna(sep_str):
        return None
    try:
        # literal_eval turns the single-quoted Python-style string
        # into a real list[dict] without executing any code
        grape_list = ast.literal_eval(sep_str)  # type: ignore
        if isinstance(grape_list, list) and grape_list:
            return grape_list[0].get("seo_name")  # dominant grape
    except (ValueError, SyntaxError):
        pass
    return None


# Add a new column containing only the first cépage name
dfv["cepages"] = dfv["cepages"].apply(extract_dominant)

dfv = dfv.dropna(subset=["cepages"])  # Remove rows with NaN in 'cepages'

# Ensure 'vintage' is numeric
dfv["vintage"] = pd.to_numeric(dfv["vintage"], errors="coerce")

# in dfr, add a year column with years from 2010 to 2024 for every region
# the regions are repeated for each year
years = pd.DataFrame({"year": range(2010, 2025)})  # 2025 excluded

# --------------------------------------------
# 3) CROSS-JOIN   (requires pandas ≥ 1.2)
# --------------------------------------------
dfr_expanded = dfr.merge(years, how="cross")

# Result: len(regions) * 15 rows
print(dfr_expanded.shape)  # e.g. (600, 2)

# --------------------------------------------
# 4) If you need to keep other per-region columns
#    (e.g. latitude, longitude, cluster_id, …)
#    merge them back:
# --------------------------------------------
dfr_expanded = dfr_expanded.merge(dfr.drop_duplicates("region"), on="region", how="left")

# delete year_x, year, latitude_x, latitude_y
dfr_expanded = dfr_expanded.drop(columns=["latitude_x", "longitude_x"])

# Rename columns year_y in year, latitude_y in latitude, longitude_y in longitude
dfr_expanded = dfr_expanded.rename(
    columns={"year_y": "year", "latitude_y": "latitude", "longitude_y": "longitude"}
)

R_EARTH_KM = 6371.0  # Earth radius


def attach_weather(
    regions: pd.DataFrame, station_by_year: dict[int, pd.DataFrame], max_km: float = 40.0
) -> pd.DataFrame:
    """
    For every (region, year) row, append the meteo values of the closest
    station from the same year lying within `max_km`.  Returns a copy of
    `regions` with new columns.  Weather frames MUST contain at least:
        ['station', 'year', 'latitude', 'longitude', ... other meteo cols].
    """

    # Pre-build one BallTree per year in *radians* --------------------------
    trees = {}
    coords = {}  # {year: ndarray([[lat, lon], ...])}
    weather = {}  # {year: df_year with index aligned to coords}
    for yr, df in station_by_year.items():
        # Drop rows missing coordinates
        dfc = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        if dfc.empty:
            continue
        xy_rad = np.radians(dfc[["latitude", "longitude"]].values)
        trees[yr] = BallTree(xy_rad, metric="haversine")
        coords[yr] = xy_rad
        weather[yr] = dfc  # keep same order as coords

    # Prepare result container --------------------------------------------
    out_rows = []

    # Process by year to avoid mixing seasons -----------------------------
    for yr, sub in regions.groupby("year"):
        if yr not in trees:  # no station data for that year
            out_rows.append(sub.assign(dist_km=np.nan))
            continue

        tree = trees[yr]
        xy_r = np.radians(sub[["latitude", "longitude"]].values)
        # k=1 ⇒ nearest neighbour; returns dist in *radians*
        dist_rad, idx = tree.query(xy_r, k=1)
        dist_km = dist_rad.flatten() * R_EARTH_KM

        # Any distance > max_km → treat as “no station”
        valid = dist_km <= max_km
        met_cols = [
            c for c in weather[yr].columns if c not in ("latitude", "longitude", "year")
        ]  # keep station & metrics
        # Build a block with NaNs then fill the valid rows
        met_block = pd.DataFrame(np.nan, index=sub.index, columns=met_cols)
        met_block.loc[valid, :] = weather[yr].iloc[idx[valid, 0]][met_cols].values

        merged = (
            sub.reset_index(drop=True)
            .join(met_block.reset_index(drop=True))
            .assign(dist_km=dist_km)
        )  # keep distance for QC
        out_rows.append(merged)

    # Concatenate every year back together ---------------------------------
    return pd.concat(out_rows, ignore_index=True)


# 1) Put your df10 … df24 into a dictionary
station_frames = {
    2010: df10,
    2011: df11,
    2012: df12,
    2013: df13,
    2014: df14,
    2015: df15,
    2016: df16,
    2017: df17,
    2018: df18,
    2019: df19,
    2020: df20,
    2021: df21,
    2022: df22,
    2023: df23,
    2024: df24,
}

dfr["latitude"] = pd.to_numeric(dfr["latitude"], errors="coerce")
dfr["longitude"] = pd.to_numeric(dfr["longitude"], errors="coerce")

# 2) Run the attachment
regions_weather = attach_weather(dfr_expanded, station_frames, max_km=40)

# rename dfv column vintage to year
dfv = dfv.rename(columns={"vintage": "year"})

# merge dfv with regions_weather on region and year
dfv = dfv.merge(regions_weather, on=["region", "year"], how="left")
# dfv now contains the weather data for each wine region and year


# number of elements in dfv
print(f"Number of elements in dfv: {dfv.shape[0]} rows, {dfv.shape[1]} columns")

# delete elements with NaN in 'cepages' and 'year' and 'hot_days_y'
dfv = dfv.dropna(subset=["cepages", "year", "hot_days"])

# number of elements in dfv after dropping NaN
print(f"Number of elements in dfv after dropping NaN: {dfv.shape[0]} rows, {dfv.shape[1]} columns")

# save dfv to parquet
dfv.to_parquet("data/vivino_wines_with_weather.parquet", index=False)
# save dfv to csv
dfv.to_csv("data/vivino_wines_with_weather.csv", index=False)


# VISUALIZATION

# plot the distance distribution
plt.figure(figsize=(10, 6))
plt.hist(dfv["dist_km"], bins=50, color="blue", alpha=0.7, edgecolor="black")
plt.title("Distance to Closest Weather Station")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.xlim(0, 35)  # limit x-axis to 100 km for better visibility
plt.grid(axis="y", alpha=0.75)
plt.show()
