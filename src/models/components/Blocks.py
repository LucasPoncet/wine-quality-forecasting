"""
Reusable neural network building blocks for tabular and deep learning models.
Includes dense, convolutional, transformer, and attention-based modules.
"""

import math

import numpy as np
import torch
import torch.nn as nn

# Dense and Convolutional Blocks


class DenseBlock(nn.Module):
    """Fully connected block with optional BatchNorm and dropout."""

    def __init__(self, in_size, out_size, activation, batch_normalization=False, dropout_rate=0.1):
        super().__init__()

        self.linear_layer = nn.Linear(in_size, out_size)
        self.activation = activation

        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm1d(out_size)
        else:
            self.batch_norm_layer = None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Apply linear transformation, normalization, activation, and dropout."""
        x = self.linear_layer(x)
        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x


class FlattenDenseBlock(nn.Module):
    """Flatten input tensor and apply a DenseBlock."""

    def __init__(self, in_size, out_size, activation, batch_normalization=False, dropout_rate=0.1):
        super().__init__()
        in_size_flatten = int(np.prod(in_size))
        self.flatten_layer = nn.Flatten()
        self.dense_layer = DenseBlock(
            in_size=in_size_flatten,
            out_size=out_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x


class Conv2DBlock(nn.Module):
    """2D convolutional block with optional BatchNorm, activation, and dropout."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        batch_normalization=False,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.activation = activation
        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm_layer = None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout_layer(x)
        return x


class BasicResNetBlock(nn.Module):
    """Residual convolutional block using two Conv2DBlock layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        batch_normalization=False,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.conv_layer1 = Conv2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )
        self.conv_layer2 = Conv2DBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Conv2DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                batch_normalization=batch_normalization,
                dropout_rate=dropout_rate,
            )

    def forward(self, x):
        residual = x
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        residual = self.shortcut(residual)
        x += residual
        return x


class UnflattenDenseBlock(nn.Module):
    """Dense block followed by unflattening to a target shape."""

    def __init__(self, in_size, out_size, activation, batch_normalization=False, dropout_rate=0.1):
        super().__init__()
        self.dense_layer = DenseBlock(
            in_size=in_size,
            out_size=np.prod(out_size),
            activation=activation,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )
        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=out_size)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.unflatten_layer(x)
        return x


# Transformer and Attention Blocks


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding as used in Transformer models."""

    pe: torch.Tensor

    def __init__(self, num_embeddings, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(num_embeddings, d_model)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register positional embedding as non-trainable parameters
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)
        return x

    def plot_positional_embedding(self):
        """Visualize positional embeddings."""
        import matplotlib.pyplot as plt
        import numpy as np

        pe_array: np.ndarray = self.pe.detach().cpu().numpy()

        plt.figure(figsize=(12, 8))
        plt.imshow(pe_array.T, aspect="auto", cmap="viridis")  # Transpose for better visualization
        plt.colorbar()
        plt.title("Positional Embedding Visualization")
        plt.xlabel("Position Index")
        plt.ylabel("Embedding Dimension")
        plt.show()


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention and MLP."""

    def __init__(
        self,
        input_dim,
        num_heads,
        expansion_factor: int = 2,
        activation: type[nn.Module] | None = None,
        dropout_rate=0.0,
    ):
        act: nn.Module = activation() if activation is not None else nn.ReLU()
        super().__init__()

        self.mha_layer = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate
        )
        self.attention_weights = None

        self.norm_layer1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        hidden_dim = input_dim * expansion_factor
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), act, nn.Linear(hidden_dim, input_dim)
        )
        self.norm_layer2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_out, self.attention_weights = self.mha_layer(x, x, x)  # (B, S, D) -> (B, S, D)
        # Add and Norm
        attention_out = attention_out + x
        out = self.norm_layer1(attention_out)  # (B, S, D)
        out = self.dropout1(out)  # (B, S, D)
        # MLP part
        ff_out = self.mlp(out)  # (B, S, D)
        out = ff_out + out
        out = self.norm_layer2(out)  # (B, S, D)
        out = self.dropout2(out)  # (B, S, D)
        return out


# TabNet-style Blocks


class GhostBatchNorm(nn.Module):
    """BatchNorm over smaller virtual batches (as in TabNet)."""

    def __init__(self, input_dim, virtual_batch_size=64, momentum=0.01):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, x):
        if not self.training or x.size(0) <= self.virtual_batch_size:
            return self.bn(x)
        chunks = x.chunk(x.size(0) // self.virtual_batch_size, dim=0)
        res = [self.bn(c) for c in chunks]
        return torch.cat(res, dim=0)


class FeatureTransformerBlock(nn.Module):
    """Stacked GLU-style feature transformation block (TabNet)."""

    def __init__(
        self, input_dim, output_dim, n_glu_layers=2, dropout_rate=0.2, virtual_batch_size=64
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_glu_layers):
            in_dim = input_dim if i == 0 else output_dim
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, output_dim),
                    GhostBatchNorm(output_dim, virtual_batch_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
            )
        self.skip = input_dim == output_dim
        self.residual = nn.Identity() if self.skip else nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = x
        for blocks in self.blocks:
            out = blocks(out)
        out = out + self.residual(x)
        return out


class Sparsemax(nn.Module):
    """Sparsemax activation (Martins & Astudillo, 2016)."""

    def forward(self, input):
        dim = -1
        input = input - input.max(dim=dim, keepdim=True)[0]

        z_sorted, _ = torch.sort(input, dim=dim, descending=True)
        k = torch.arange(1, input.size(dim) + 1, device=input.device).view(1, -1)
        k = k.expand_as(z_sorted)

        z_cumsum = z_sorted.cumsum(dim)
        support = (1 + k * z_sorted) > z_cumsum

        k_max = support.sum(dim=dim, keepdim=True)

        tau_sum = z_cumsum.gather(dim, k_max - 1)
        tau = (tau_sum - 1) / k_max.float()

        output = torch.clamp(input - tau, min=0)
        return output


class AttentiveTransformer(nn.Module):
    """TabNet-style attentive transformer producing sparse attention masks."""

    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.01):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size, momentum)
        self.sparsemax = Sparsemax()

    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        x = self.sparsemax(x)
        return x * prior
