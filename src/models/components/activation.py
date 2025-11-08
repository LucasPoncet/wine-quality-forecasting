from torch import nn


def get_activation(name: str | None) -> nn.Module | None:
    """Return a PyTorch activation module by name.

    Parameters
    ----------
    name : str | None
        Activation name (e.g., 'relu', 'sigmoid', 'tanh', 'linear' or None).

    Returns
    -------
    nn.Module | None
        The corresponding activation function, or None for linear activation.
    """
    if not name:
        return None
    match name.lower():
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "gelu":
            return nn.GELU()
        case "leakyrelu":
            return nn.LeakyReLU()
        case "elu":
            return nn.ELU()
        case "linear":
            return None
        case _:
            raise ValueError(f"Unsupported activation name: {name}")
