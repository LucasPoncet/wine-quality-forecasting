import torch

NUMERIC_FEATURES = {
    "A": ("heat_to_rain_ratio", ["GDD", "rain_June", "rain_SepOct"]),
    "B": ("diurnal_range_summer", ["TX_summer", "TM_summer"]),
    "C": ("hot_day_intensity", ["hot_days", "TX_summer"]),
    "D": ("frost_risk_index", ["frost_days_Apr", "avg_TM_Apr"]),
    "E": ("price_per_heat_unit", ["price", "GDD"]),
    "I": ("log_price", ["price"]),
}

CATEGORICAL_FEATURES = {
    "J": ("region_station_id", ["region", "station"]),
}


def compute_numeric_feature(fid: str, data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute engineered numeric feature by ID."""
    if fid == "A":
        return data["GDD"] / (data["rain_June"] + data["rain_SepOct"] + 1.0)
    if fid == "B":
        return data["TX_summer"] - data["TM_summer"]
    if fid == "C":
        return data["hot_days"] / (data["TX_summer"] + 1.0)
    if fid == "D":
        return data["frost_days_Apr"] / (data["avg_TM_Apr"] + 1.0)
    if fid == "E":
        return data["price"] / (data["GDD"] + 1.0)
    if fid == "I":
        return torch.log1p(data["price"])
    raise ValueError(f"Unsupported numeric feature: {fid}")


def compute_categorical_feature(fid: str, data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute engineered categorical feature by ID."""
    if fid == "J":
        return data["region"] * 1000 + data["station"]
    raise ValueError(f"Unsupported categorical feature: {fid}")
