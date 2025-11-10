import pytest
import torch

from src.models.architectures.ft_transformer import FeatureTokenizer, FTTransformer

# Fixtures


@pytest.fixture
def embedding_sizes():
    """Dummy categorical embedding sizes (name → (vocab_size, emb_dim))."""
    return {
        "region": (5, 3),
        "station": (4, 2),
        "cepages": (3, 2),
    }


@pytest.fixture
def hyperparameters():
    """Minimal hyperparameter dictionary for FTTransformer."""
    return {
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "dropout_rate": 0.1,
        "activation": "relu",
        "output_dim": 3,
    }


@pytest.fixture
def dummy_input(embedding_sizes):
    """Return small random tensors for numeric and categorical input."""
    x_num = torch.randn(8, 4)  # 8 samples, 4 numeric features
    x_cat = torch.stack(
        [torch.randint(0, vocab_size, (8,)) for vocab_size, _ in embedding_sizes.values()], dim=1
    )
    return x_num, x_cat


# FeatureTokenizer Tests


def test_tokenizer_output_shape(embedding_sizes):
    tokenizer = FeatureTokenizer(
        num_numeric_features=4, categorical_sizes=embedding_sizes, d_model=8
    )
    x_num = torch.randn(10, 4)
    x_cat = torch.stack([torch.randint(0, v, (10,)) for v, _ in embedding_sizes.values()], dim=1)
    tokens = tokenizer(x_num, x_cat)
    # Expect 10 samples, 4 + 3 = 7 features, each 8-dimensional
    assert tokens.shape == (10, 7, 8)


def test_tokenizer_gradients(embedding_sizes):
    tokenizer = FeatureTokenizer(
        num_numeric_features=2, categorical_sizes=embedding_sizes, d_model=8
    )
    x_num = torch.randn(5, 2, requires_grad=True)
    x_cat = torch.stack([torch.randint(0, v, (5,)) for v, _ in embedding_sizes.values()], dim=1)
    tokens = tokenizer(x_num, x_cat)
    loss = tokens.sum()
    loss.backward()
    # Ensure gradients exist for numeric affine parameters
    assert tokenizer.a.grad is not None
    assert tokenizer.b.grad is not None


def test_tokenizer_no_categorical():
    tokenizer = FeatureTokenizer(num_numeric_features=3, categorical_sizes={}, d_model=8)
    x_num = torch.randn(6, 3)
    x_cat = torch.empty(6, 0, dtype=torch.long)
    out = tokenizer(x_num, x_cat)
    assert out.shape == (6, 3, 8)


# FTTransformer Tests


def test_forward_output_shape(hyperparameters, embedding_sizes, dummy_input):
    model = FTTransformer(hyperparameters, embedding_sizes, num_numeric_features=4)
    x_num, x_cat = dummy_input
    logits = model(x_num, x_cat)
    assert logits.shape == (8, hyperparameters["output_dim"])
    assert torch.isfinite(logits).all()


def test_multiple_layers_forward_pass(hyperparameters, embedding_sizes, dummy_input):
    """Ensure multi-layer stack does not change shape unexpectedly."""
    hyper = hyperparameters.copy()
    hyper["n_layers"] = 3
    model = FTTransformer(hyper, embedding_sizes, num_numeric_features=4)
    x_num, x_cat = dummy_input
    logits = model(x_num, x_cat)
    assert logits.shape == (8, hyper["output_dim"])


def test_gradients_flow(hyperparameters, embedding_sizes, dummy_input):
    """Check that gradients propagate through the entire network."""
    model = FTTransformer(hyperparameters, embedding_sizes, num_numeric_features=4)
    x_num, x_cat = dummy_input
    y_pred = model(x_num, x_cat)
    loss = y_pred.sum()
    loss.backward()

    grads_exist = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grads_exist, "No gradient flow detected through FTTransformer."


def test_device_transfer(hyperparameters, embedding_sizes, dummy_input):
    """Ensure model can move to CUDA (if available) and still run."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FTTransformer(hyperparameters, embedding_sizes, num_numeric_features=4).to(device)
    x_num, x_cat = [t.to(device) for t in dummy_input]
    out = model(x_num, x_cat)
    assert out.device == device


def test_dropout_effect(hyperparameters, embedding_sizes, dummy_input):
    """Output distributions should differ slightly when dropout > 0."""
    hyper = hyperparameters.copy()
    hyper["dropout_rate"] = 0.5
    model = FTTransformer(hyper, embedding_sizes, num_numeric_features=4)
    x_num, x_cat = dummy_input

    model.train()
    out_train = model(x_num, x_cat)
    model.eval()
    out_eval = model(x_num, x_cat)

    diff = torch.mean(torch.abs(out_train - out_eval)).item()
    assert diff > 0 or hyper["dropout_rate"] == 0.0


def test_eval_consistency(hyperparameters, embedding_sizes, dummy_input):
    """Model should produce identical outputs in eval mode."""
    model = FTTransformer(hyperparameters, embedding_sizes, num_numeric_features=4)
    x_num, x_cat = dummy_input
    model.eval()
    y1 = model(x_num, x_cat)
    y2 = model(x_num, x_cat)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_forward_requires_proper_shapes(hyperparameters, embedding_sizes):
    """Mismatched feature counts should raise a runtime error."""
    model = FTTransformer(hyperparameters, embedding_sizes, num_numeric_features=4)
    x_num = torch.randn(8, 2)  # wrong number of numeric features
    x_cat = torch.stack([torch.randint(0, v, (8,)) for v, _ in embedding_sizes.values()], dim=1)
    with pytest.raises(RuntimeError):
        _ = model(x_num, x_cat)


def test_tokenizer_output_backprop_integration(hyperparameters, embedding_sizes, dummy_input):
    """Integrated test combining tokenizer + transformer for gradient backprop."""
    model = FTTransformer(hyperparameters, embedding_sizes, num_numeric_features=4)
    x_num, x_cat = dummy_input
    logits = model(x_num, x_cat)
    loss = logits.mean()
    loss.backward()
    assert all(p.grad is not None for p in model.head.parameters())
