import numpy as np

from src.models.architectures.tabular_mlp import TabularMLP


def make_mlp_model(hyperparams, cat_cols, mapping=None):
    if not cat_cols or mapping is None:
        embedding_sizes = {}
    else:
        embedding_sizes = {
            col: (len(mapping[col]), int(max(4, np.sqrt(len(mapping[col])) // 2)))
            for col in cat_cols
        }
    model = TabularMLP(hyperparams, embedding_sizes)
    return model, embedding_sizes
