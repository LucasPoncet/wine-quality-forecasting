import torch.nn as nn

from src.models.components.activation import get_activation
from src.models.components.blocks import DenseBlock


class MLP(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()

        self.hidden_layers_size = hyperparameters.get("hidden_layers_size", [64, 32])
        self.activation = get_activation(hyperparameters.get("activation", "relu"))
        self.batch_normalization = hyperparameters.get("batch_normalization", False)
        self.dropout_rate = hyperparameters.get("dropout_rate", 0.0)
        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]

        self.layers = nn.ModuleList()

        layer = DenseBlock(
            in_size=self.input_dim,
            out_size=self.hidden_layers_size[0],
            activation=self.activation,
            batch_normalization=self.batch_normalization,
            dropout_rate=self.dropout_rate,
        )
        self.layers.append(layer)
        self.n_dense_layer = len(self.hidden_layers_size)
        for i in range(self.n_dense_layer - 1):
            layer = DenseBlock(
                in_size=self.hidden_layers_size[i],
                out_size=self.hidden_layers_size[i + 1],
                activation=self.activation,
                batch_normalization=self.batch_normalization,
                dropout_rate=self.dropout_rate,
            )
            self.layers.append(layer)
        layer = nn.Linear(in_features=self.hidden_layers_size[-1], out_features=self.output_dim)
        self.layers.append(layer)
        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        """Run input through the sequential MLP classifier."""
        x_hat = self.classifier(x)
        return x_hat
