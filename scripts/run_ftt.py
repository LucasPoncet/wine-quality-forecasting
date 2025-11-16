from pathlib import Path

import torch

from src.models.training.ftt_runner import run_ftt_pipeline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ftt_pipeline(
        train_path=Path("data/vivino_wine_train_label.parquet"),
        test_path=Path("data/vivino_wine_test_label.parquet"),
        feature_ids=["A", "B", "C"],
        device=device,
        max_epoch=200,
        plot=True,
    )


if __name__ == "__main__":
    main()
