import torch

from src.models.architectures.ft_transformer import FTTransformer
from src.models.builders.ftt_builder import make_ftt_model


def test_make_ftt_model_builds_correctly():
    """Ensure make_ftt_model returns a valid FTTransformer and embedding sizes."""
    # --- Mock inputs ---
    hyperparams = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "activation": "gelu",
        "dropout_rate": 0.1,
        "output_dim": 2,
        "learning_rate": 1e-3,
        "max_epoch": 10,
    }
    num_cols = ["f1", "f2", "f3"]
    cat_cols = ["region", "station"]
    onehot_mapping = {
        "region": {"A": 0, "B": 1, "C": 2},
        "station": {"S1": 0, "S2": 1},
    }

    # --- Run builder ---
    model, embedding_sizes = make_ftt_model(hyperparams, num_cols, cat_cols, onehot_mapping)

    # --- Assertions ---
    assert isinstance(model, FTTransformer)
    assert isinstance(embedding_sizes, dict)
    assert all(isinstance(v, tuple) and isinstance(v[0], int) for v in embedding_sizes.values())
    assert set(embedding_sizes.keys()) == set(cat_cols)

    # Ensure model has expected numeric and categorical setup
    assert model.num_numeric_features == len(num_cols)
    assert hasattr(model, "forward")

    # --- Run forward pass sanity check ---
    x_num = torch.randn(4, len(num_cols))
    x_cat = torch.zeros((4, len(cat_cols)), dtype=torch.long)
    out = model(x_num, x_cat)
    assert out.shape[0] == 4
