import pytest
import torch

from src.models.architectures.tabnet import TabNetClassifier, TabNetEncoder

# ------------------------ FIXTURES -----------------------------


@pytest.fixture
def embedding_sizes():
    """Returns embedding sizes dict used by TabNetClassifier."""
    return {"region": (5, 3), "station": (4, 2), "cepages": (3, 2)}


@pytest.fixture
def dummy_input(embedding_sizes):
    """Returns simple random numeric and categorical inputs."""
    x_num = torch.randn(8, 4)  # 8 samples, 4 numeric features
    x_cat = torch.stack(
        [torch.randint(0, vocab, (8,)) for vocab, _ in embedding_sizes.values()], dim=1
    )  # 3 categorical features
    return x_num, x_cat


@pytest.fixture
def encoder():
    """A lightweight encoder for testing."""
    return TabNetEncoder(
        input_dim=10,
        output_dim=8,
        n_steps=3,
        n_shared=1,
        n_independent=1,
        virtual_batch_size=4,
    )


# ---------------------- TabNetEncoder --------------------------


def test_encoder_output_shape(encoder):
    """Ensure TabNetEncoder returns correct output shape."""
    x = torch.randn(5, encoder.input_dim)
    out = encoder(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, encoder.output_dim)


def test_encoder_masks_shape(encoder):
    """Check that masks are returned and shaped properly."""
    x = torch.randn(5, encoder.input_dim)
    out, masks = encoder(x, return_masks=True)

    assert isinstance(out, torch.Tensor)
    assert isinstance(masks, list)
    assert len(masks) == encoder.n_steps - 1

    for mask in masks:
        assert mask.shape == (5, encoder.input_dim)
        assert torch.all(mask >= 0) and torch.all(mask <= 1)


def test_encoder_prior_behavior(encoder):
    """Ensure masks reduce the prior (monotonic decrease)."""
    x = torch.randn(6, encoder.input_dim)
    _, masks = encoder(x, return_masks=True)

    priors = []
    prior = torch.ones_like(x)
    for mask in masks:
        new_prior = prior * (1 - mask).clamp_min(1e-5)
        priors.append(new_prior)
        prior = new_prior

    # Prior norms should be decreasing
    norms = [p.norm().item() for p in priors]
    assert all(a >= b for a, b in zip(norms, norms[1:], strict=False)), (
        "Prior should monotonically decrease."
    )


def test_encoder_requires_grad_flow(encoder):
    """Ensure gradients flow through TabNetEncoder."""
    x = torch.randn(4, encoder.input_dim, requires_grad=True)
    y = encoder(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_encoder_reset_masks(encoder):
    """Check that masks are reset between forward calls."""
    x = torch.randn(4, encoder.input_dim)
    _, masks1 = encoder(x, return_masks=True)
    _, masks2 = encoder(x, return_masks=True)
    # Object identity should differ
    assert masks1 is not masks2
    assert len(encoder.collected_masks) == encoder.n_steps - 1


# -------------------- TabNetClassifier ------------------------


def test_classifier_output_shape(embedding_sizes, dummy_input):
    """Ensure classifier outputs correct logits shape."""
    x_num, x_cat = dummy_input
    model = TabNetClassifier(
        embedding_sizes=embedding_sizes,
        num_numeric_features=x_num.shape[1],
        output_dim=3,
        n_steps=3,
        shared_layers=1,
        step_layers=1,
    )

    out = model(x_num, x_cat)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (x_num.size(0), 3)


def test_classifier_with_masks(embedding_sizes, dummy_input):
    """Ensure return_masks=True yields both logits and masks."""
    x_num, x_cat = dummy_input
    model = TabNetClassifier(
        embedding_sizes=embedding_sizes,
        num_numeric_features=x_num.shape[1],
        output_dim=2,
        n_steps=4,
        shared_layers=1,
        step_layers=1,
    )

    logits, masks = model(x_num, x_cat, return_masks=True)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(masks, list)
    assert logits.shape[0] == x_num.size(0)
    assert all(mask.ndim == 2 for mask in masks)


def test_classifier_gradients(embedding_sizes, dummy_input):
    """Check full gradient flow through classifier."""
    x_num, x_cat = dummy_input
    model = TabNetClassifier(embedding_sizes, num_numeric_features=4, output_dim=2)
    y_pred = model(x_num, x_cat)
    loss = y_pred.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "No gradients propagated through the model."
    for g in grads:
        assert torch.isfinite(g).all()


def test_classifier_dropout_effect(embedding_sizes, dummy_input):
    """Ensure dropout layer behaves stochastically."""
    x_num, x_cat = dummy_input
    model = TabNetClassifier(embedding_sizes, num_numeric_features=4, emb_dropout=0.5)
    model.train()

    # two forward passes should differ (stochastic dropout)
    out1 = model(x_num, x_cat)
    out2 = model(x_num, x_cat)
    assert not torch.allclose(out1, out2), "Dropout should introduce stochasticity."


def test_classifier_eval_consistency(embedding_sizes, dummy_input):
    """Ensure deterministic output in eval mode."""
    x_num, x_cat = dummy_input
    model = TabNetClassifier(embedding_sizes, num_numeric_features=4, emb_dropout=0.5)
    model.eval()

    out1 = model(x_num, x_cat)
    out2 = model(x_num, x_cat)
    assert torch.allclose(out1, out2), "Model in eval mode should produce consistent outputs."


def test_classifier_device_transfer(embedding_sizes, dummy_input):
    """Ensure model can move between CPU and CUDA (if available)."""
    x_num, x_cat = dummy_input
    model = TabNetClassifier(embedding_sizes, num_numeric_features=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out = model(x_num.to(device), x_cat.to(device))
    assert out.device == device


def test_invalid_input_shape_raises(embedding_sizes):
    """Passing mismatched categorical input should raise."""
    model = TabNetClassifier(embedding_sizes, num_numeric_features=4)
    x_num = torch.randn(8, 4)
    # Wrong number of categorical columns
    x_cat = torch.randint(0, 5, (8, 2))
    with pytest.raises(IndexError):
        _ = model(x_num, x_cat)
