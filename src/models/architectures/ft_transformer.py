import logging

import torch
import torch.nn as nn

from src.models.components.activation import get_activation
from src.models.components.blocks import (
    TransformerEncoderBlock,
)

logger = logging.getLogger(__name__)


# ---------- 1.  Feature-tokeniser -------------------------------------
class FeatureTokenizer(nn.Module):
    """
    Converts numeric and categorical features into fixed-size embedding tokens.

    Each feature (numeric or categorical) becomes a d_model-dimensional vector token.

    - Numeric: τ_k = a_k + b_k · x_k  (learned affine transform per feature)
    - Categorical: standard nn.Embedding

    Parameters
    ----------
    num_numeric_features : int
        Number of continuous features.
    categorical_sizes : dict[str, tuple[int, int]]
        Dictionary of categorical feature names and (vocab_size, embedding_dim).
        The embedding_dim is ignored; all categorical features use d_model.
    d_model : int
        Dimension of token embeddings (Transformer hidden size).
    """

    def __init__(
        self, num_numeric_features: int, categorical_sizes: dict[str, tuple[int, int]], d_model: int
    ):
        super().__init__()

        self.num_numeric_features = num_numeric_features
        self.d_model = d_model

        # Numeric affine parameters: τ_k = a_k + b_k·x_k
        self.a = nn.Parameter(torch.zeros(num_numeric_features, d_model))
        self.b = nn.Parameter(torch.ones(num_numeric_features, d_model))

        # Categorical embeddings: one embedding table per categorical feature
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, d_model)
                for name, (vocab_size, _dim) in categorical_sizes.items()
            }
        )

        logger.debug(
            "FeatureTokenizer initialized with %d numeric and %d categorical features",
            num_numeric_features,
            len(categorical_sizes),
        )

    def forward(self, x_num, x_cat):
        """Return concatenated numeric and categorical feature tokens (B, F_total, D)."""
        # Numeric tokens
        num_tokens = self.a + self.b * x_num.unsqueeze(-1)  # (B, Fnum, D)

        # Categorical tokens
        cat_tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings.values())]
        cat_tokens = (
            torch.stack(cat_tokens, dim=1) if cat_tokens else torch.empty_like(num_tokens[:, :0, :])
        )

        # Concatenate
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        return tokens


# ---------- 2.  FT-Transformer encoder -------------------------------
class FTTransformer(nn.Module):
    """
    Feature Token Transformer (FT-Transformer)
    ------------------------------------------
    Applies Transformer self-attention across feature tokens (not sequence tokens).

    The model learns inter-feature dependencies using standard Transformer
    encoder blocks and outputs class logits from the [CLS] token.

    References
    ----------
    - Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (2021)

    Parameters
    ----------
    hyper : dict
        Dictionary of model hyperparameters:
            - d_model : int
            - n_layers : int
            - n_heads : int
            - dropout_rate : float
            - activation : str
            - output_dim : int
    embedding_sizes : dict[str, tuple[int, int]]
        Categorical feature name → (vocab_size, embedding_dim) mapping.
    num_numeric_features : int
        Number of numeric input features.
    """

    def __init__(
        self, hyper: dict, embedding_sizes: dict[str, tuple[int, int]], num_numeric_features: int
    ):
        super().__init__()
        self.d_model = hyper["d_model"]
        self.n_classes = hyper["output_dim"]
        self.n_layers = hyper["n_layers"]
        self.n_heads = hyper["n_heads"]
        self.dropout_rate = hyper["dropout_rate"]
        self.activation = get_activation(hyper["activation"])

        # 1. feature tokeniser
        self.tokenizer = FeatureTokenizer(
            num_numeric_features=num_numeric_features,
            categorical_sizes=embedding_sizes,
            d_model=self.d_model,
        )

        # 2. CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 3. stack of encoder blocks
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    input_dim=self.d_model,
                    num_heads=self.n_heads,
                    activation=self.activation,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.n_layers)
            ]
        )

        # 4. classifier head
        self.head = nn.Linear(self.d_model, self.n_classes)
        logger.info(
            "FTTransformer initialized with %d layers, %d heads, d_model=%d",
            self.n_layers,
            self.n_heads,
            self.d_model,
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass: feature tokenization → attention → classification."""
        batch_size = x_num.size(0)

        # Tokenize features
        x = self.tokenizer(x_num, x_cat)  # (B, F, D)

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + F, D)

        # Transformer encoder expects (S, B, D)
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.transpose(0, 1)

        # Use CLS token output for classification
        cls_output = x[:, 0, :]
        logits = self.head(cls_output)
        return logits
