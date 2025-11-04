import torch
from torch.utils.data import TensorDataset

_NUM_FEATS = {
    "A": ("heat_to_rain_ratio", ["GDD", "rain_June", "rain_SepOct"]),
    "B": ("diurnal_range_summer", ["TX_summer", "TM_summer"]),
    "C": ("hot_day_intensity", ["hot_days", "TX_summer"]),
    "D": ("frost_risk_index", ["frost_days_Apr", "avg_TM_Apr"]),
    "E": ("price_per_heat_unit", ["price", "GDD"]),
    "I": ("log_price", ["price"]),
}

_CAT_FEATS = {
    "J": ("region_station_id", ["region", "station"]),
}


def _compute_feature(letter: str, data_row: dict[str, torch.Tensor]):
    """Return a scalar tensor for the requested engineered feature."""
    if letter == "A":
        return data_row["GDD"] / (data_row["rain_June"] + data_row["rain_SepOct"] + 1.0)
    if letter == "B":
        return data_row["TX_summer"] - data_row["TM_summer"]
    if letter == "C":
        return data_row["hot_days"] / (data_row["TX_summer"] + 1.0)
    if letter == "D":
        return data_row["frost_days_Apr"] / (data_row["avg_TM_Apr"] + 1.0)
    if letter == "E":
        return data_row["price"] / (data_row["GDD"] + 1.0)
    if letter == "I":
        return torch.log1p(data_row["price"])
    raise ValueError(f"Unsupported numeric feature ID: {letter}")


def _compute_cat_feature(letter: str, data_row: dict[str, torch.Tensor]):
    if letter == "J":
        return data_row["region"] * 1000 + data_row["station"]
    raise ValueError(f"Unsupported categorical feature ID: {letter}")


def add_engineered_features(
    datasets: tuple[TensorDataset, ...],
    num_cols: list[str],
    cat_cols: list[str],
    feature_ids: list[str],
) -> tuple[tuple[TensorDataset, ...], list[str], list[str]]:
    """Append engineered features to each TensorDataset.

    Parameters
    ----------
    datasets      : tuple of TensorDataset (train, valid, test, …)
    num_cols      : list of existing numeric column names (order matches tensor)
    cat_cols      : list of existing categorical column names
    feature_ids   : list like ["B", "A", "D"] choosing which engineered
                     features to add.

    Returns
    -------
    new_datasets  : tuple of new TensorDataset with augmented tensors
    new_num_cols  : updated numeric column name list
    new_cat_cols  : updated categorical column name list
    """
    num_ids = [f for f in feature_ids if f in _NUM_FEATS]
    cat_ids = [f for f in feature_ids if f in _CAT_FEATS]

    num_index = {name: idx for idx, name in enumerate(num_cols)}
    cat_index = {name: idx for idx, name in enumerate(cat_cols)}

    new_datasets = []
    for ds in datasets:
        x_num, x_cat, y = ds.tensors
        data_dict = {name: x_num[:, num_index[name]] for name in num_cols}
        data_dict.update({name: x_cat[:, cat_index[name]].long() for name in cat_cols})

        new_num_feats = []
        for letter in num_ids:
            feat_tensor = _compute_feature(letter, data_dict).unsqueeze(1)
            new_num_feats.append(feat_tensor)
        if new_num_feats:
            x_num = torch.cat([x_num] + new_num_feats, dim=1)

        new_cat_feats = []
        for letter in cat_ids:
            cat_tensor = _compute_cat_feature(letter, data_dict).unsqueeze(1)
            new_cat_feats.append(cat_tensor)
        if new_cat_feats:
            x_cat = torch.cat([x_cat] + new_cat_feats, dim=1)

        new_datasets.append(TensorDataset(x_num, x_cat, y))

    new_num_cols = num_cols + [_NUM_FEATS[idx][0] for idx in num_ids]
    new_cat_cols = cat_cols + [_CAT_FEATS[idx][0] for idx in cat_ids]

    return tuple(new_datasets), new_num_cols, new_cat_cols


def drop_columns(
    datasets: tuple[TensorDataset, ...],
    num_cols: list[str],
    drop_list: list[str],
) -> tuple[tuple[TensorDataset, ...], list[str]]:
    """
    Remove the numeric columns in `drop_list` from every TensorDataset.
    Categorical columns are left untouched.
    """
    keep_idx = [i for i, name in enumerate(num_cols) if name not in drop_list]
    new_num_cols = [num_cols[i] for i in keep_idx]

    new_datasets = []
    for ds in datasets:
        x_num, x_cat, y = ds.tensors
        x_num = x_num[:, keep_idx]
        new_datasets.append(TensorDataset(x_num, x_cat, y))

    return tuple(new_datasets), new_num_cols
