from src.models.architectures.tabular_mlp import TabularMLP


def build_model(hyperparameters: dict, embedding_sizes: dict):
    model_type = hyperparameters.get("model_type", "mlp").lower()
    if model_type == "mlp":
        return TabularMLP(hyperparameters, embedding_sizes)
    # elif model_type == "fttransformer":
    #     return FTTransformer(hyperparameters, embedding_sizes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
