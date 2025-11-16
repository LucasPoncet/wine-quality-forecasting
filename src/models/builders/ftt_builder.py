from collections.abc import Mapping
from typing import Any

import numpy as np

from src.models.architectures.ft_transformer import FTTransformer


def make_ftt_model(
    hyperparams: dict[str, Any],
    num_cols: list[str],
    cat_cols: list[str],
    onehot_mapping: Mapping[str, Mapping[str, int] | list[Any]],
):
    """
    Build an FT-Transformer model and its embedding sizes from categorical vocabularies.

    Parameters
    ----------
    hyperparams : dict
        Dictionary of FTTransformer hyperparameters.
    num_cols : list[str]
        List of numerical feature names.
    cat_cols : list[str]
        List of categorical feature names.
    onehot_mapping : dict[str, list]
        Mapping categorical columns → vocabulary indices.

    Returns
    -------
    model : FTTransformer
        Instantiated FT-Transformer model.
    embedding_sizes : dict[str, tuple[int, int | None]]
        Dictionary of (vocab_size, embed_dim) per categorical feature.
    """
    # Compute (vocab_size, embed_dim)
    embedding_sizes: dict[str, tuple[int, int]] = {
        col: (len(onehot_mapping[col]), max(8, int(np.sqrt(len(onehot_mapping[col])))))
        for col in cat_cols
        if col in onehot_mapping
    }

    # Instantiate model
    model = FTTransformer(
        hyper=hyperparams,
        embedding_sizes=embedding_sizes,
        num_numeric_features=len(num_cols),
    )

    # Keep consistency with other architectures
    object.__setattr__(model, "num_numeric_features", len(num_cols))
    object.__setattr__(model, "cat_cols", cat_cols)
    object.__setattr__(model, "num_cols", num_cols)
    object.__setattr__(model, "embedding_sizes", embedding_sizes)

    return model, embedding_sizes
