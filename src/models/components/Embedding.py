import logging

import pandas as pd
import torch

from src.utils.config_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def build_cat_mapping(
    data: pd.DataFrame | dict,
    cat_cols: list[str],
    existing: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, dict[str, int]], torch.Tensor, list[int]]:
    """Encode categorical columns into integer tensors with persistent mappings.

    Parameters
    ----------
    data : pd.DataFrame | dict
        Input data containing categorical columns.
    cat_cols : list[str]
        Names of categorical columns to encode.
    existing : dict[str, dict[str, int]] | None, optional
        Optional existing mapping `{col: {category: index}}`
        to ensure consistent encoding across datasets.

    Returns
    -------
    mapping : dict[str, dict[str, int]]
        Updated mapping of category → index for each column.
    x_cat : torch.Tensor
        Tensor of shape (n_rows, n_cat_features) with integer-encoded categories.
    vocab_sizes : list[int]
        Number of unique values per categorical column.

    Raises
    ------
    KeyError
        If a column in `cat_cols` is missing from the input data.
    """
    df = pd.DataFrame(data) if isinstance(data, dict) else data.copy()

    mapping: dict[str, dict[str, int]] = existing or {}
    encoded_cols = []
    vocab_sizes = []

    for col in cat_cols:
        if col not in df.columns:
            raise KeyError(f"Categorical column '{col}' not found in data.")
        if col not in mapping:
            mapping[col] = {}

        col_map = mapping[col]
        codes = []
        for val in df[col].astype(str):
            if val not in col_map:
                col_map[val] = len(col_map)
            codes.append(col_map[val])
        encoded_cols.append(torch.tensor(codes, dtype=torch.long).unsqueeze(1))
        vocab_sizes.append(len(col_map))
        logger.debug("Encoded column '%s' with %d unique values", col, len(col_map))

    x_cat = (
        torch.cat(encoded_cols, dim=1)
        if encoded_cols
        else torch.empty((len(df), 0), dtype=torch.long)
    )
    return mapping, x_cat, vocab_sizes
