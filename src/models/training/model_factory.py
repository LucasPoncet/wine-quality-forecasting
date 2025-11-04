from ClassesML.TabularMLP import TabularMLP


def build_model(hyperparameters, embedding_sizes):
    return TabularMLP(hyperparameters, embedding_sizes)
