import ast  # For parsing the 'cepages' string safely
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept": "application/json",
}

BASE_PARAMS = {
    "country_code": "FR",
    "country_codes[]": "fr",
    "currency_code": "EUR",
    "grape_filter": "varietal",
    "min_rating": "1",
    "order_by": "price",
    "order": "asc",
    "price_range_min": "0",
    "price_range_max": "500000",
    "wine_type_ids[]": "1",
}


def scrape_all_vivino_pages(pmin, pmax):
    url = "https://www.vivino.com/api/explore/explore"
    all_results = []

    # Step 1: Request page 1 to find out total number of pages
    params = BASE_PARAMS.copy()
    params["page"] = 1

    params["price_range_min"] = pmin
    params["price_range_max"] = pmax

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Initial request failed with status code {response.status_code}")

    data = response.json()
    total_wines = data["explore_vintage"]["records_matched"]
    wines_per_page = len(data["explore_vintage"]["matches"])
    total_pages = math.ceil(total_wines / wines_per_page)

    print(f"🔍 Found {total_wines} wines across {total_pages} pages.")

    # Step 2: Loop through all pages
    for page in range(1, total_pages + 1):
        print(f"📄 Scraping page {page}/{total_pages}")
        params["page"] = page
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"⚠️ Skipping page {page} due to error {response.status_code}")
            continue

        matches = response.json()["explore_vintage"]["matches"]
        for t in matches:
            try:
                all_results.append(
                    (
                        t["vintage"]["wine"]["winery"]["name"],
                        f"{t['vintage']['wine']['name']} {t['vintage']['year']}",
                        t["vintage"]["year"],
                        t["vintage"]["statistics"]["ratings_average"],
                        t["vintage"]["statistics"]["ratings_count"],
                        t["vintage"]["statistics"]["wine_ratings_average"],
                        t["vintage"]["statistics"]["wine_ratings_count"],
                        t["vintage"]["wine"]["region"]["seo_name"],
                        t["price"]["amount"],
                        t["vintage"]["wine"]["style"]["grapes"],
                    )
                )
            except Exception as e:
                print("X Skipping a wine due to missing data:", e)

        time.sleep(0.5)  # Be polite to the server

    return pd.DataFrame(
        all_results,
        columns=[
            "Winery",
            "Wine",
            "vintage",
            "vintage_rating",
            "vintage_rating_count",
            "wine_rating",
            "wine_rating_count",
            "region",
            "price",
            "cepages",
        ],
    )


# lauch the scraper

df0 = scrape_all_vivino_pages(
    0, 50000
)  # Vivino doesn't allow us to scrap to many wines at once so we have to split the scraping in several intervals that follow the prices
df1 = scrape_all_vivino_pages(15, 500000)
df2 = scrape_all_vivino_pages(21.8, 500000)
df3 = scrape_all_vivino_pages(30.1, 500000)
df4 = scrape_all_vivino_pages(42.1, 500000)
df5 = scrape_all_vivino_pages(62.1, 500000)
df6 = scrape_all_vivino_pages(107.1, 500000)
df7 = scrape_all_vivino_pages(288.1, 500000)


# Combine all dataframes into one and clean it up
df_combined = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["Wine", "vintage"], keep="first")
df_combined = df_combined.sort_values(by="price")


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
        grape_list = ast.literal_eval(sep_str)
        if isinstance(grape_list, list) and grape_list:
            return grape_list[0].get("seo_name")  # dominant grape
    except (ValueError, SyntaxError):
        pass
    return None


# Add a new column containing only the first cépage name
df_combined["cepages"] = df_combined["cepages"].apply(extract_dominant)

df_combined = df_combined.dropna(subset=["cepages"])  # Remove rows with NaN in 'cepages'

# Ensure 'vintage' is numeric
df_combined["vintage"] = pd.to_numeric(df_combined["vintage"], errors="coerce")
vintage_counts = df_combined["vintage"].value_counts().sort_index()

df_combined.to_parquet(
    "wines_vivino.parquet", index=False
)  # Save the cleaned DataFrame to a Parquet file


# VISUALIZATION


# Plotting the distribution of wines by vintage
plt.figure(figsize=(12, 5))
vintage_counts.plot(kind="bar", color="skyblue")
plt.title("Number of Wines per Vintage")
plt.xlabel("Vintage Year")
plt.ylabel("Number of Wines")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis="y")
plt.show()


# TOP 90% REGIONS

# Count wines per region
region_counts = df_combined["region"].value_counts()

# Compute cumulative wine share per region
cumulative_share = region_counts.cumsum() / region_counts.sum()

# Keep only regions that cumulatively make up the top 90%
top_90_mask = cumulative_share <= 0.9
top_90_regions = region_counts[top_90_mask]

# Get the ordered list of those region names (most frequent to least within top 90%)
ordered_top_regions = top_90_regions.index.tolist()

#  Output the list
print(" Top 90% Regions (ordered by wine count):")
for region in ordered_top_regions:
    print("-", region)

# === Optional Plot ===
plt.figure(figsize=(12, 6))
top_90_regions.sort_values(ascending=True).plot(kind="barh", color="mediumseagreen")
plt.title("Regions Representing Top 90% of Wines")
plt.xlabel("Number of Wines")
plt.ylabel("Region")
plt.grid(axis="x")
plt.tight_layout()
plt.show()
