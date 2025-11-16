"""
Main entry point for adding engineered features to tabular TensorDatasets.
Builds on the numeric and categorical feature definitions from
`src.preprocessing.utils.feature_utils`.

Example
-------
new_datasets, new_num_cols, new_cat_cols = add_engineered_features(
    (train_ds, valid_ds, test_ds),
    num_cols,
    cat_cols,
    feature_ids=["A", "B", "J"],
)
"""

import logging

import torch
from torch.utils.data import TensorDataset

from src.preprocessing.utils.feature_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    compute_categorical_feature,
    compute_numeric_feature,
)

logger = logging.getLogger()


def add_engineered_features(
    datasets: tuple[TensorDataset, ...],
    num_cols: list[str],
    cat_cols: list[str],
    feature_ids: list[str],
) -> tuple[tuple[TensorDataset, ...], list[str], list[str]]:
    """
    Append engineered numeric and categorical features to a set of TensorDatasets.

    Parameters
    ----------
    datasets : tuple[TensorDataset, ...]
        Datasets (train, valid, test, etc.), each containing (x_num, x_cat, y) tensors.
    num_cols : list[str]
        Names of existing numeric columns.
    cat_cols : list[str]
        Names of existing categorical columns.
    feature_ids : list[str]
        Feature IDs to engineer, e.g. ["A", "B", "J"].

    Returns
    -------
    new_datasets : tuple[TensorDataset, ...]
        Datasets with augmented tensors.
    new_num_cols : list[str]
        Updated numeric column names.
    new_cat_cols : list[str]
        Updated categorical column names.
    """
    num_ids = [fid for fid in feature_ids if fid in NUMERIC_FEATURES]
    cat_ids = [fid for fid in feature_ids if fid in CATEGORICAL_FEATURES]

    num_index = {name: i for i, name in enumerate(num_cols)}
    cat_index = {name: i for i, name in enumerate(cat_cols)}

    new_datasets = []

    for ds in datasets:
        x_num, x_cat, y = ds.tensors

        # Prepare lookup dict
        data_row = {name: x_num[:, num_index[name]] for name in num_cols}
        data_row.update({name: x_cat[:, cat_index[name]].long() for name in cat_cols})

        # Compute engineered numeric features
        new_num_feats = [compute_numeric_feature(fid, data_row).unsqueeze(1) for fid in num_ids]
        if new_num_feats:
            x_num = torch.cat([x_num] + new_num_feats, dim=1)

        # Compute engineered categorical features
        new_cat_feats = [compute_categorical_feature(fid, data_row).unsqueeze(1) for fid in cat_ids]
        if new_cat_feats:
            x_cat = torch.cat([x_cat] + new_cat_feats, dim=1)

        new_datasets.append(TensorDataset(x_num, x_cat, y))

    new_num_cols = num_cols + [NUMERIC_FEATURES[i][0] for i in num_ids]
    new_cat_cols = cat_cols + [CATEGORICAL_FEATURES[i][0] for i in cat_ids]

    logger.info(
        "Added %d numeric and %d categorical engineered features",
        len(num_ids),
        len(cat_ids),
    )

    return tuple(new_datasets), new_num_cols, new_cat_cols


def drop_columns(
    datasets: tuple[TensorDataset, ...],
    num_cols: list[str],
    drop_list: list[str],
) -> tuple[tuple[TensorDataset, ...], list[str]]:
    """
    Remove specific numeric columns from each TensorDataset.

    Parameters
    ----------
    datasets : tuple[TensorDataset, ...]
        Datasets containing (x_num, x_cat, y).
    num_cols : list[str]
        list of numeric feature names.
    drop_list : list[str]
        Names of numeric columns to drop.

    Returns
    -------
    new_datasets : tuple[TensorDataset, ...]
        Same datasets with selected columns removed.
    new_num_cols : list[str]
        Remaining numeric column names.
    """
    keep_idx = [i for i, name in enumerate(num_cols) if name not in drop_list]
    new_num_cols = [num_cols[i] for i in keep_idx]

    new_datasets = []
    for ds in datasets:
        x_num, x_cat, y = ds.tensors
        x_num = x_num[:, keep_idx]
        new_datasets.append(TensorDataset(x_num, x_cat, y))

    logger.info("Dropped %d numeric columns", len(drop_list))
    return tuple(new_datasets), new_num_cols
