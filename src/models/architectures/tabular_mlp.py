import torch
import torch.nn as nn

from src.models.components.activation import get_activation
from src.models.components.blocks import DenseBlock


class TabularMLP(nn.Module):
    def __init__(self, hyperparameters, embedding_sizes):
        super().__init__()

        self.hidden_layers_size = hyperparameters.get("hidden_layers_size", [128, 64])
        self.activation = get_activation(hyperparameters.get("activation", "relu"))
        self.batch_normalization = hyperparameters.get("batch_normalization", False)
        self.dropout_rate = hyperparameters.get("dropout_rate", 0.1)
        num_numeric_features = hyperparameters.get("num_numeric_features", 0)
        output_dim = hyperparameters.get("output_dim", 1)

        self.emb_layers = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in embedding_sizes.items()}
        )
        emb_total_dim = sum(dim for (_, dim) in embedding_sizes.values()) if embedding_sizes else 0

        total_input_dim = num_numeric_features + emb_total_dim
        layers: list[nn.Module] = []
        layers.append(
            DenseBlock(
                in_size=total_input_dim,
                out_size=self.hidden_layers_size[0],
                activation=self.activation,
                batch_normalization=self.batch_normalization,
                dropout_rate=self.dropout_rate,
            )
        )
        for in_size, out_size in zip(
            self.hidden_layers_size[:-1], self.hidden_layers_size[1:], strict=False
        ):
            layers.append(
                DenseBlock(
                    in_size=in_size,
                    out_size=out_size,
                    activation=self.activation,
                    batch_normalization=self.batch_normalization,
                    dropout_rate=self.dropout_rate,
                )
            )
        layers.append(nn.Linear(self.hidden_layers_size[-1], output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_numeric: torch.Tensor, x_cat: torch.Tensor):
        """Concatenate numeric features and (optional) embeddings then classify."""
        if x_cat is not None and self.emb_layers:
            emb_tensors = [
                layer(x_cat[:, idx]) for idx, layer in enumerate(self.emb_layers.values())
            ]
            x = torch.cat([x_numeric] + emb_tensors, dim=1)
        else:
            x = x_numeric
        return self.classifier(x)
