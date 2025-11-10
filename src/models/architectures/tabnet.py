from typing import cast

import torch
import torch.nn as nn

from src.models.components.blocks import AttentiveTransformer, FeatureTransformerBlock


class TabNetEncoder(nn.Module):
    """Core encoder block of TabNet: sequential feature selection and transformation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_steps: int = 5,
        n_shared: int = 2,
        n_independent: int = 2,
        virtual_batch_size: int = 128,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.shared_feat_transform = nn.ModuleList(
            [
                FeatureTransformerBlock(
                    input_dim, output_dim, virtual_batch_size=virtual_batch_size
                )
                for _ in range(n_shared)
            ]
        )

        self.step_feat_transform = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FeatureTransformerBlock(
                            output_dim, output_dim, virtual_batch_size=virtual_batch_size
                        )
                        for _ in range(n_independent)
                    ]
                )
                for _ in range(n_steps)
            ]
        )

        self.attentive_transformers = nn.ModuleList(
            [
                AttentiveTransformer(output_dim, input_dim, virtual_batch_size)
                for _ in range(n_steps)
            ]
        )

        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.collected_masks: list[torch.Tensor] = []

    def forward(self, x, return_masks=False):
        """Forward pass through all TabNet steps."""
        x = self.initial_bn(x)
        prior = torch.ones_like(x)
        outputs = []
        masked_x = x
        self.collected_masks = []
        outputs = []

        for step in range(self.n_steps):
            out = masked_x

            # Shared transformation
            for block in self.shared_feat_transform:
                out = block(out)

            # Step-specific transformation
            step_blocks: nn.ModuleList = cast(nn.ModuleList, self.step_feat_transform[step])
            for block in step_blocks:
                out = block(out)

            outputs.append(out)

            if step < self.n_steps - 1:
                mask = self.attentive_transformers[step](out, prior)
                self.collected_masks.append(mask)
                masked_x = mask * x
                prior = prior * (1 - mask).clamp_min(1e-5)

        if return_masks:
            return torch.stack(outputs, 0).sum(0), self.collected_masks
        return torch.stack(outputs, dim=0).sum(dim=0)


class TabNetClassifier(nn.Module):
    """TabNet classifier variant combining embeddings, encoder, and output layer."""

    def __init__(
        self,
        embedding_sizes: dict[str, tuple[int, int]],
        num_numeric_features: int,
        output_dim: int = 2,
        n_steps: int = 5,
        shared_layers: int = 2,
        step_layers: int = 2,
        emb_dropout: float = 0.0,
        virtual_batch_size: int = 128,
    ):
        super().__init__()

        # Build embedding layers
        self.emb_layers = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in embedding_sizes.items()}
        )

        emb_total_dim = sum(dim for _, dim in embedding_sizes.values())
        self.emb_dropout = nn.Dropout(emb_dropout)

        total_input_dim = num_numeric_features + emb_total_dim

        self.encoder = TabNetEncoder(
            input_dim=total_input_dim,
            output_dim=total_input_dim,
            n_steps=n_steps,
            n_shared=shared_layers,
            n_independent=step_layers,
            virtual_batch_size=virtual_batch_size,
        )

        self.output_layer = nn.Linear(total_input_dim, output_dim)

    def forward(self, x_num, x_cat, return_masks=False):
        """Forward pass through TabNet classifier."""
        emb_tensors = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers.values())]
        x = torch.cat([x_num] + emb_tensors, dim=1)
        x = self.emb_dropout(x)

        features = self.encoder(x)
        if return_masks:
            features, masks = self.encoder(x, return_masks=True)
            return self.output_layer(features), masks
        logits = self.output_layer(features)
        return logits
