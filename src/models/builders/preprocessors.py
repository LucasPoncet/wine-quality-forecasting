from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUM_FEATURES: list[str] = [
    "GDD",
    "TM_summer",
    "TX_summer",
    "temp_amp_summer",
    "hot_days",
    "rainy_days_summer",
    "rain_June",
    "rain_SepOct",
    "frost_days_Apr",
    "avg_TM_Apr",
    "price",
]

CAT_FEATURES: list[str] = ["cepages", "region"]


def _categorical_encoder(kind: str):
    if kind == "linear":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _make_preprocessor(model_key: str) -> ColumnTransformer:
    if model_key == "lr":
        transformers = [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
                ),
                NUM_FEATURES,
            ),
            ("cat", _categorical_encoder("linear"), CAT_FEATURES),
        ]
    elif model_key == "xgb":
        transformers = [
            ("num", SimpleImputer(strategy="mean"), NUM_FEATURES),
            ("cat", _categorical_encoder("tree"), CAT_FEATURES),
        ]
    else:
        transformers = [
            ("num", SimpleImputer(strategy="mean"), NUM_FEATURES),
            ("cat", _categorical_encoder("tree"), CAT_FEATURES),
        ]
    return ColumnTransformer(transformers)
