from collections.abc import Mapping
from typing import Any

import numpy as np

from src.models.architectures.tabnet import TabNetClassifier


def make_tabnet_model(
    hyperparams: dict[str, Any],
    num_cols: list[str],
    cat_cols: list[str],
    onehot_mapping: Mapping[str, Mapping[str, int] | list[Any]],
):
    """
    Build a TabNet model and embedding sizes from categorical vocabularies.

    Parameters
    ----------
    hyperparams : dict
        Dictionary of TabNet hyperparameters.
    num_cols : list[str]
        List of numerical feature names.
    cat_cols : list[str]
        List of categorical feature names.
    onehot_mapping : dict[str, list]
        Dictionary mapping categorical columns to their vocabulary indices.

    Returns
    -------
    model : TabNetClassifier
        Instantiated TabNet model.
    embedding_sizes : dict[str, tuple[int, int]]
        Dictionary of (vocab_size, embed_dim) per categorical feature.
    """
    embedding_sizes = {
        col: (len(onehot_mapping[col]), max(8, int(np.sqrt(len(onehot_mapping[col])))))
        for col in cat_cols
        if col in onehot_mapping
    }

    model = TabNetClassifier(
        embedding_sizes=embedding_sizes,
        num_numeric_features=len(num_cols),
        output_dim=2,
        n_steps=hyperparams.get("n_steps", 5),
        shared_layers=hyperparams.get("shared_layers", 2),
        step_layers=hyperparams.get("step_layers", 2),
        emb_dropout=hyperparams.get("emb_dropout", 0.0),
        virtual_batch_size=hyperparams.get("virtual_batch", 128),
    )

    return model, embedding_sizes
