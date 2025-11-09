import logging
import warnings

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.utils.config_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and preprocess tabular wine data for deep learning models.

    This class handles:
      - Loading parquet data for train/test (and optional valid split)
      - Numerical normalization and NaN imputation
      - Categorical mapping to integer indices
      - TensorDataset creation for PyTorch

    Parameters
    ----------
    train_path : str
        Path to the parquet file containing the training (and optionally validation) data.
    test_path : str
        Path to the parquet file containing the test data.
    target_col : str, default="label"
        Target column name for classification/regression.
    num_cols : list[str] | None, default=None
        Names of numerical feature columns.
    cat_cols : list[str] | None, default=None
        Names of categorical feature columns.
    valid_frac : float, default=0.2
        Fraction of training data to hold out for validation if no explicit 'split' column.
    dtype : torch.dtype, default=torch.float32
        Tensor dtype for numerical features.
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        target_col: str = "label",
        num_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
        valid_frac: float = 0.2,
        dtype: torch.dtype = torch.float32,
    ):
        self.train_path, self.test_path, self.target_col = train_path, test_path, target_col
        self.num_cols, self.cat_cols = num_cols, cat_cols
        self.valid_frac, self.dtype = valid_frac, dtype
        self.cat_mapping: dict[str, dict[str, int]] | None = None

    def create_index_mapping(self, df, categorical_columns):
        mapping = {}
        for col in categorical_columns:
            unique_vals = sorted(df[col].dropna().unique())
            mapping[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return mapping

    def _clean_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replace NaNs/Infs and clamp large values."""
        t = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.clamp(t, -1e6, 1e6)
        return t

    def _df_to_dataset(self, df: pd.DataFrame) -> TensorDataset:
        """Convert a DataFrame to a TensorDataset with numerical & categorical tensors."""
        df = df.copy()

        # Numeric features
        if self.num_cols:
            for col in self.num_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].mean())
            x_num = torch.as_tensor(df[self.num_cols].to_numpy(), dtype=self.dtype)
            x_num = self._clean_tensor(x_num)
        else:
            x_num = torch.empty((len(df), 0), dtype=self.dtype)

        # Categorical features
        if self.cat_cols:
            for col in self.cat_cols:
                df[col] = df[col].fillna("__MISSING__")

            if self.cat_mapping is None:
                raise ValueError("cat_mapping must be initialized before encoding.")

            cat_encoded = {
                col: df[col].map(self.cat_mapping[col]).fillna(0).astype(int)
                for col in self.cat_cols
            }
            x_cat = torch.as_tensor(pd.DataFrame(cat_encoded).to_numpy(), dtype=torch.long)
        else:
            x_cat = torch.empty((len(df), 0), dtype=torch.long)

        # Target
        y_series = pd.to_numeric(df[self.target_col], errors="coerce").fillna(0)
        y = torch.tensor(y_series.values, dtype=torch.long)

        return TensorDataset(x_num, x_cat, y)

    def load_tabular_data(
        self,
    ) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, dict[str, int]], int]:
        """Load parquet datasets and return PyTorch TensorDatasets for training.

        Returns
        -------
        tuple
            (train_ds, valid_ds, test_ds, cat_mapping, n_classes)
        """
        logger.info("Loading train data from %s", self.train_path)
        train_valid = pd.read_parquet(self.train_path)
        logger.info("Loading test data from %s", self.test_path)
        test = pd.read_parquet(self.test_path)

        if self.num_cols is None and self.cat_cols:
            excluded = set(self.cat_cols + [self.target_col])
            self.num_cols = [c for c in train_valid.columns if c not in excluded]
        elif self.cat_cols is None and self.num_cols is None:
            raise ValueError("At least one of num_cols or cat_cols must be provided.")

        if "split" in train_valid.columns:
            train_df = train_valid[train_valid["split"] == "train"].copy()
            valid_df = train_valid[train_valid["split"] == "valid"].copy()
        else:
            train_df = train_valid.sample(frac=1 - self.valid_frac, random_state=100)
            valid_df = train_valid.drop(train_df.index)

        if self.cat_cols:
            self.cat_mapping = self.create_index_mapping(train_valid, self.cat_cols)
        else:
            self.cat_mapping = {}

        train_ds = self._df_to_dataset(train_df)
        valid_ds = self._df_to_dataset(valid_df)
        test_ds = self._df_to_dataset(test)

        n_classes = int(train_valid[self.target_col].nunique())
        logger.info("Data prepared: %d classes detected", n_classes)
        return train_ds, valid_ds, test_ds, self.cat_mapping, n_classes
