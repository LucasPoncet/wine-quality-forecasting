from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import BarContainer


def plot_model_comparison(metrics_dict: dict[str, dict[str, float]], save_path: str | None = None):
    """
    Plot comparison of model metrics automatically for any number of models.

    Parameters
    ----------
    metrics_dict : dict
        Example:
        {
            "MLP": {"Accuracy": 0.82, "F1": 0.77, "Precision 0": 0.92, "Recall 0": 0.79},
            "LGBM": {"Accuracy": 0.78, "F1": 0.71, "Precision 0": 0.87, "Recall 0": 0.77},
            "FTT": {"Accuracy": 0.81, "F1": 0.76, "Precision 0": 0.91, "Recall 0": 0.83},
        }
    save_path : str | None
        If provided, saves the plot as a PNG to this path.
    """
    if not isinstance(metrics_dict, dict):
        raise ValueError("metrics_dict must be a dict[str, dict[str, float]]")

    # Data prep

    df = pd.DataFrame(metrics_dict).T  # models → rows, metrics → columns
    metrics = df.columns
    models = df.index

    x = np.arange(len(metrics))
    n_models = len(models)
    width = 0.8 / n_models  # auto width based on number of models

    # Plot

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        ax.bar(
            x + (i - n_models / 2) * width + width / 2,
            df.loc[model],
            width,
            label=model,
        )

    # Labels & Formatting

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.legend(frameon=True)
    ax.set_ylim(0, 1.05)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved plot to {save_path}")

    plt.show()
