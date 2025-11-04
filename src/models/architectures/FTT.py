# ClassesML/FTTransformer.py
import torch, torch.nn as nn
from Utils.Utilities          import Utilities        # ← already used elsewhere
from ClassesML.Blocks         import TransformerEncoderBlock          # :contentReference[oaicite:0]{index=0}

# ---------- 1.  Feature-tokeniser -------------------------------------
class FeatureTokenizer(nn.Module):
    """
    Turns every (numeric or categorical) feature into a d_model-dim token.
    numeric:  τ_k = a_k + b_k·x_k          (learned scale & bias)
    categorical: standard nn.Embedding.
    """
    def __init__(self, num_num_features: int,
                       cat_sizes: dict[str, tuple[int,int]],
                       d_model: int):
        super().__init__()
        # numeric → token parameters (a, b) per feature
        self.a = nn.Parameter(torch.zeros(num_num_features, d_model))
        self.b = nn.Parameter(torch.ones (num_num_features, d_model))

        # categorical embeddings
        self.cat_embs = nn.ModuleDict({
            name: nn.Embedding(vocab, d_model)
            for name, (vocab, _d) in cat_sizes.items()
        })

    def forward(self, x_num, x_cat):
        # numeric
        num_tokens = self.a + self.b * x_num.unsqueeze(-1)      # (B, Fnum, D)

        # categorical
        cat_tokens = []
        for i, name in enumerate(self.cat_embs.keys()):
            cat_tokens.append(self.cat_embs[name](x_cat[:, i])) # list[(B, D)]
        cat_tokens = torch.stack(cat_tokens, dim=1)             # (B, Fcat, D)

        return torch.cat([num_tokens, cat_tokens], dim=1)       # (B, F, D)

# ---------- 2.  FT-Transformer encoder -------------------------------
class FTTransformer(nn.Module):
    def __init__(self, hyper, embedding_sizes, num_numeric_features):
        super().__init__()
        self.d_model       = hyper["d_model"]        # e.g. 64
        self.n_classes     = hyper["output_dim"]
        self.n_layers      = hyper["n_layers"]       # e.g. 4
        self.n_heads       = hyper["n_heads"]        # e.g. 8
        self.dropout_rate  = hyper["dropout_rate"]

        # 2.1 feature tokeniser
        self.tokenizer = FeatureTokenizer(
            num_num_features   = num_numeric_features,
            cat_sizes          = embedding_sizes,
            d_model            = self.d_model,
        )

        # 2.2 CLS token (learnable)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 2.3 stack of encoder blocks
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(
                input_dim       = self.d_model,
                num_heads       = self.n_heads,
                activation      = Utilities.get_activation(hyper["activation"]),
                dropout_rate    = self.dropout_rate,
            )
            for _ in range(self.n_layers)
        ])

        # 2.4 classifier head
        self.head = nn.Linear(self.d_model, self.n_classes)

    # ------------------------------ #
    def forward(self, x_num, x_cat):                    # (B, Fnum)  (B, Fcat)
        B = x_num.size(0)

        # tokens
        x = self.tokenizer(x_num, x_cat)                # (B, F, D)
        cls = self.cls.expand(B, -1, -1)                # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # prepend CLS → (B, 1+F, D)

        # transformer expects (S, B, D)
        x = x.transpose(0,1)
        for block in self.encoder:
            x = block(x)                                # still (S, B, D)
        x = x.transpose(0,1)                            # back to (B, S, D)

        cls_out = x[:, 0, :]                            # (B, D)
        return self.head(cls_out)                      # logits
