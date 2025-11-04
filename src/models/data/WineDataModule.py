import pandas as pd
import torch
from torch.utils.data import TensorDataset


class DatasetLoader:
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

    def load_tabular_data(self):
        train_valid = pd.read_parquet(self.train_path)
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

        self.cat_mapping = self.create_index_mapping(train_valid, self.cat_cols)

        def df_to_ds(df: pd.DataFrame) -> TensorDataset:
            def _clean(t: torch.Tensor) -> torch.Tensor:
                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.clamp(t, -1e6, 1e6)
                return t

            df = df.copy()

            if self.num_cols:
                for col in self.num_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].mean())
                x_num = torch.tensor(df[self.num_cols].values, dtype=self.dtype)
                x_num = _clean(x_num)
            else:
                x_num = torch.empty((len(df), 0), dtype=self.dtype)
            if self.cat_cols:
                for col in self.cat_cols:
                    df[col] = df[col].fillna("__MISSING__")

            x_cat = df[self.cat_cols].copy()
            if self.cat_cols:
                assert self.cat_mapping is not None  # protection
                for col in self.cat_cols:
                    x_cat[col] = x_cat[col].map(self.cat_mapping[col]).fillna(0).astype(int)

            x_cat = torch.tensor(x_cat.values, dtype=torch.long)

            # Labels
            y_series = pd.to_numeric(df[self.target_col], errors="coerce").fillna(0)
            y = torch.tensor(y_series.values, dtype=torch.long)

            return TensorDataset(x_num, x_cat, y)

        train_ds = df_to_ds(train_df)
        valid_ds = df_to_ds(valid_df)
        test_ds = df_to_ds(test)

        n_classes = int(train_valid[self.target_col].nunique())
        return train_ds, valid_ds, test_ds, self.cat_mapping, n_classes
