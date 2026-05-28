import os

import pytest
import torch
from src.trukan.trukan import TruKan
from src.trukan.trukan_plotter import TruKanPlotter
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


def test_trukan_saved(setup_trukan, setup_dataset):
    model = setup_trukan
    dataset = setup_dataset

    # model(setup_dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "trukan_saved"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()

    assert os.path.isfile("./figures/trukan_saved.svg")
