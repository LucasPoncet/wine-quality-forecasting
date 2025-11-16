import pandas as pd
import pytest

from src.preprocessing.merge_wine_weather import (
    extract_dominant_cepage,
    merge_wine_and_weather,
)

# 1. UNIT TESTS — extract_dominant_cepage


@pytest.mark.parametrize(
    "input_val,expected",
    [
        ("[{'seo_name': 'merlot'}, {'seo_name': 'cabernet'}]", "merlot"),
        ("[{'seo_name': 'chardonnay'}]", "chardonnay"),
        ("[]", None),
        ("not a list", None),
        (float("nan"), None),
        ("[{'wrong_key': 'nope'}]", None),
    ],
)
def test_extract_dominant_cepage_various_inputs(input_val, expected):
    assert extract_dominant_cepage(input_val) == expected


# 2. INTEGRATION TESTS — merge_wine_and_weather


def test_merge_wine_and_weather_pipeline(monkeypatch, tmp_path):
    """Test the end-to-end pipeline with mocked weather attachment and plotting."""

    # --- Create dummy CSV inputs ---
    regions_csv = tmp_path / "regions.csv"
    vivino_csv = tmp_path / "vivino.csv"
    output_dir = tmp_path / "out"
    weather_dir = tmp_path / "weather"

    pd.DataFrame(
        {
            "region": ["Bordeaux", "Champagne"],
            "latitude": [44.8, 49.0],
            "longitude": [-0.6, 3.9],
        }
    ).to_csv(regions_csv, index=False)

    pd.DataFrame(
        {
            "region": ["Bordeaux", "Champagne"],
            "vintage": [2020, 2021],
            "cepages": ["[{'seo_name': 'merlot'}]", "[{'seo_name': 'pinot noir'}]"],
            "hot_days": [5, 6],  # include to avoid NaN filtering
        }
    ).to_csv(vivino_csv, index=False)

    # --- Mock weather attachment ---
    def fake_attach_weather_pandas(regions_expanded, weather_dir, max_km):
        df = regions_expanded.copy()
        df["dist_km"] = 12.0
        return df

    monkeypatch.setattr(
        "src.preprocessing.merge_wine_weather.attach_weather_pandas",
        fake_attach_weather_pandas,
    )

    # --- Mock plotting ---
    monkeypatch.setattr(
        "src.preprocessing.merge_wine_weather.plot_histogram",
        lambda *a, **k: None,
    )

    # --- Run the merge ---
    merged = merge_wine_and_weather(
        regions_csv=regions_csv,
        vivino_csv=vivino_csv,
        weather_dir=weather_dir,
        output_dir=output_dir,
        max_km=50.0,
        year_range=(2020, 2022),
    )

    # --- Assertions ---
    assert isinstance(merged, pd.DataFrame)
    assert "region" in merged.columns
    assert "year" in merged.columns
    assert "hot_days" in merged.columns
    assert output_dir.exists()
    assert (output_dir / "vivino_wines_with_weather.csv").exists()
    assert (output_dir / "vivino_wines_with_weather.parquet").exists()


def test_merge_wine_and_weather_handles_empty_cepages(monkeypatch, tmp_path):
    """Ensure that rows with NaN 'cepages' are dropped gracefully."""
    regions_csv = tmp_path / "regions.csv"
    vivino_csv = tmp_path / "vivino.csv"
    output_dir = tmp_path / "out"
    weather_dir = tmp_path / "weather"

    pd.DataFrame({"region": ["Bordeaux"], "latitude": [45.0], "longitude": [0.0]}).to_csv(
        regions_csv, index=False
    )

    # include NaN in cepages
    pd.DataFrame({"region": ["Bordeaux"], "vintage": [2020], "cepages": [float("nan")]}).to_csv(
        vivino_csv, index=False
    )

    monkeypatch.setattr(
        "src.preprocessing.merge_wine_weather.attach_weather_pandas",
        lambda *a, **k: pd.DataFrame({"region": ["Bordeaux"], "year": [2020], "hot_days": [10]}),
    )
    monkeypatch.setattr("src.preprocessing.merge_wine_weather.plot_histogram", lambda *a, **k: None)

    merged = merge_wine_and_weather(
        regions_csv=regions_csv,
        vivino_csv=vivino_csv,
        weather_dir=weather_dir,
        output_dir=output_dir,
    )

    # Row with NaN cepages should be removed
    assert merged.empty or merged.shape[0] == 0
