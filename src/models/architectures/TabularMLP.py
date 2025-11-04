import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock
from Utils.Utilities import Utilities


class TabularMLP(nn.Module):
    def __init__(self, hyperparameters, embedding_sizes):
        super().__init__()

        self.hidden_layers_size = hyperparameters["hidden_layers_size"]
        self.activation = Utilities.get_activation(hyperparameters["activation"])
        self.batch_normalization = hyperparameters["batch_normalization"]
        self.dropout_rate = hyperparameters["dropout_rate"]
        num_numeric_features = hyperparameters["num_numeric_features"]
        output_dim = hyperparameters["output_dim"]

        self.emb_layers = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in embedding_sizes.items()}
        )
        emb_total_dim = sum(dim for (_, dim) in embedding_sizes.values())

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
        if self.emb_layers:
            emb_tensors = [
                layer(x_cat[:, idx]) for idx, layer in enumerate(self.emb_layers.values())
            ]
            x = torch.cat([x_numeric] + emb_tensors, dim=1)
        else:
            x = x_numeric
        return self.classifier(x)
