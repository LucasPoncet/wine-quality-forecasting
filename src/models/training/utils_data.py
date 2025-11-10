from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset


def load_parquet_dataset(train_path: Path, test_path: Path, target_col: str):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].astype(int)
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(int)
    return X_train, y_train, X_test, y_test


def clean_tensor_nan(ds: TensorDataset) -> TensorDataset:
    x_num, *rest = ds.tensors
    x_num = torch.nan_to_num(x_num, nan=0.0, posinf=0.0, neginf=0.0)
    ds.tensors = (x_num, *rest)
    return ds
