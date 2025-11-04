import torch
import torch.nn as nn
from ClassesML.Blocks import AttentiveTransformer, FeatureTransformerBlock


class TabNetEncoder(nn.Module):
    def __init__(
        self, input_dim, output_dim, n_steps=5, n_shared=2, n_independent=2, virtual_batch_size=128
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

    def forward(self, x, return_masks=False):
        x = self.initial_bn(x)
        prior = torch.ones_like(x)
        outputs = []
        masked_x = x
        self.collected_masks = []
        for step in range(self.n_steps):
            out = masked_x
            for block in self.shared_feat_transform:
                out = block(out)

            for block in self.step_feat_transform[step]:  # type:ignore
                out = block(out)

            outputs.append(out)

            if step < self.n_steps - 1:
                mask = self.attentive_transformers[step](out, prior)
                self.collected_masks.append(mask)
                masked_x = mask * x
                prior = prior * (1 - mask + 1e-5)
        if return_masks:
            return torch.stack(outputs, 0).sum(0), self.collected_masks
        return torch.stack(outputs, dim=0).sum(dim=0)


class TabNetClassifier(nn.Module):
    def __init__(
        self,
        embedding_sizes,
        num_numeric_features,
        output_dim=2,
        n_steps=5,
        shared_layers=2,
        step_layers=2,
        emb_dropout=0.0,
        virtual_batch_size=128,
    ):
        super().__init__()

        region_vocab_size, region_emb_dim = embedding_sizes["region"]
        station_vocab_size, station_emb_dim = embedding_sizes["station"]
        cepages_vocab_size, cepages_emb_dim = embedding_sizes["cepages"]

        self.region_emb = nn.Embedding(region_vocab_size, region_emb_dim)
        self.station_emb = nn.Embedding(station_vocab_size, station_emb_dim)
        self.cepages_emb = nn.Embedding(cepages_vocab_size, cepages_emb_dim)

        self.emb_dropout = nn.Dropout(emb_dropout)

        total_input_dim = num_numeric_features + region_emb_dim + station_emb_dim + cepages_emb_dim

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
        # Embedding concat
        x_emb = torch.cat(
            [
                self.region_emb(x_cat[:, 0]),
                self.station_emb(x_cat[:, 1]),
                self.cepages_emb(x_cat[:, 2]),
            ],
            dim=1,
        )

        x = torch.cat([x_num, x_emb], dim=1)
        x = self.emb_dropout(x)

        features = self.encoder(x)
        if return_masks:
            features, masks = self.encoder(x, return_masks=True)
            return self.output_layer(features), masks
        logits = self.output_layer(features)
        return logits
