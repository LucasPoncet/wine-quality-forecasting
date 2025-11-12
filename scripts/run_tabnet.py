import logging
from pathlib import Path

import torch

from src.models.training.tabnet_runner import run_tabnet_pipeline
from src.utils.config_logger import setup_logging


def main():
    setup_logging()
    logging.info("=== Starting TabNet training pipeline ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc, f1, auc = run_tabnet_pipeline(
        train_path=Path("data/vivino_wine_train_label.parquet"),
        test_path=Path("data/vivino_wine_test_label.parquet"),
        feature_ids=["A", "B", "D"],
        device=device,
        max_epoch=100,
        plot=True,
    )

    logging.info(f"Final results — Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")


if __name__ == "__main__":
    main()
