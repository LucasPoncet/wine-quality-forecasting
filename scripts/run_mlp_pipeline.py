from pathlib import Path

import torch

from src.models.training.mlp_runner import run_mlp_pipeline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_mlp_pipeline(
        train_path=Path("data/vivino_wine_train_label.parquet"),
        test_path=Path("data/vivino_wine_test_label.parquet"),
        feature_ids=["A", "B", "D"],
        device=device,
        max_epoch=1500,
    )


if __name__ == "__main__":
    main()
