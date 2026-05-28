import pytest
import torch
from src.trukan.trukan import TruKan
from src.trukan.trukan_metadata import TruKanMetadata
from src.trukan.utils import create_dataset


@pytest.fixture
def setup_dataset():
    def f(x):
        return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    yield dataset


@pytest.fixture
def setup_trukan():
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=False,
        device="cpu",
    )
    model = model.to("cpu")

    yield model


def test_network_flow(setup_trukan, setup_dataset):
    model = setup_trukan
    dataset = setup_dataset

    torch.set_default_dtype(torch.float64)

    device = "cpu"
    model = model.to(device)
    model(dataset["train_input"])

    metadata = TruKanMetadata(model)
    metadata.compute_scores(dataset["train_input"])
    metadata.plot_network_flow()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    def f(x):
        return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=False,
        device="cpu",
    )

    device = "cpu"
    model = model.to(device)
    model(dataset["train_input"])

    metadata = TruKanMetadata(model)
    metadata.compute_scores(dataset["train_input"])
    metadata.plot_layer_scores(0)
    metadata.plot_network_flow()
    print(metadata.get_feature_importance())
    print(metadata.get_top_features())
