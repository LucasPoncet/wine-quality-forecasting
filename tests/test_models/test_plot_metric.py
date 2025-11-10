import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.visualization.plots.plot_metrics import plot_model_comparison


@pytest.fixture
def sample_metrics():
    return {
        "MLP": {"Accuracy": 0.82, "F1": 0.77, "Precision 0": 0.91},
        "LGBM": {"Accuracy": 0.78, "F1": 0.71, "Precision 0": 0.87},
    }


def test_plot_runs_without_error(sample_metrics):
    """Function should execute and produce a valid matplotlib figure."""
    plot_model_comparison(sample_metrics)
    assert plt.gcf() is not None  # current figure exists


def test_plot_saves_to_file(tmp_path, sample_metrics):
    """Should correctly save PNG when save_path is provided."""
    save_path = tmp_path / "comparison.png"
    plot_model_comparison(sample_metrics, save_path=str(save_path))
    assert save_path.exists()


def test_invalid_input_raises():
    """Should raise for malformed input (e.g., non-dict)."""
    with pytest.raises(ValueError):
        plot_model_comparison("invalid")  # type: ignore[arg-type]
