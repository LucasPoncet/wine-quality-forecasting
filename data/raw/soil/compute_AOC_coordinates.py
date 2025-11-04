import geopandas as gpd
from tqdm import tqdm

# === Setup ===
tqdm.pandas()

# === Define task list ===
steps = [
    "Loading shapefile",
    "Filtering AOC wines",
    "Selecting relevant columns",
    "Dissolving polygons by AOC name",
    "Simplifying geometries",
    "Exporting to GeoJSON",
]

# === Start progress bar ===
with tqdm(
    total=len(steps),
    desc="Overall Progress",
    bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} • {desc}",
) as pbar:
    # 1. Load shapefile
    shapefile_path = "data/Soil/2025_06_10_soil_data.shp"
    gdf = gpd.read_file(shapefile_path)
    pbar.set_description(steps[0])
    pbar.update(1)

    # 2. Filter AOC wines
    aoc_only = gdf[gdf["signe"] == "AOC"]
    pbar.set_description(steps[1])
    pbar.update(1)

    # 3. Select relevant columns
    aoc_clean = aoc_only[["app", "type_prod", "categorie", "geometry"]]
    pbar.set_description(steps[2])
    pbar.update(1)

    # 4. Dissolve polygons
    aoc_dissolved = aoc_clean.dissolve(by="app", as_index=False)
    pbar.set_description(steps[3])
    pbar.update(1)

    # 5. Simplify geometries with inner progress
    pbar.set_description(steps[4])
    aoc_dissolved["geometry"] = aoc_dissolved["geometry"].progress_apply(
        lambda geom: geom.simplify(tolerance=40, preserve_topology=True)
    )
    pbar.update(1)

    # 6. Export result
    output_path = "data/Soil/aoc_polygons.geojson"
    aoc_dissolved = aoc_dissolved.to_crs(epsg=4326)
    aoc_dissolved.to_file(output_path, driver="GeoJSON")
    pbar.set_description(steps[5])
    pbar.update(1)

print(f"Saved dissolved AOC polygons to: {output_path}")
