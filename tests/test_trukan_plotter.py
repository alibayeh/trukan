import os

import pytest
import torch
from src.trukan.trukan import TruKan
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.utils import create_dataset


@pytest.fixture
def dataset():
    def f(x):
        return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    yield dataset


@pytest.fixture
def trukan_shared():
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=False,
        shared_knots=True,
        device="cpu",
    )
    model = model.to("cpu")

    yield model


@pytest.fixture
def trukan_individual():
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=False,
        shared_knots=False,
        device="cpu",
    )
    model = model.to("cpu")

    yield model


@pytest.fixture
def trukan_learn_shared():
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=True,
        shared_knots=True,
        device="cpu",
    )
    model = model.to("cpu")

    yield model


@pytest.fixture
def trukan_learn_individual():
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=True,
        shared_knots=False,
        device="cpu",
    )
    model = model.to("cpu")

    yield model


def test_trukan_shared(trukan_shared, dataset):
    model = trukan_shared
    dataset = dataset

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "trukan_shared"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()

    assert os.path.isfile("./figures/trukan_shared.svg")


def test_trukan_individual(trukan_individual, dataset):
    model = trukan_individual
    dataset = dataset

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "trukan_individual"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()

    assert os.path.isfile("./figures/trukan_individual.svg")


def test_trukan_learn_shared(trukan_learn_shared, dataset):
    model = trukan_learn_shared
    dataset = dataset

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "trukan_learn_shared"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()

    assert os.path.isfile("./figures/trukan_learn_shared.svg")


def test_trukan_learn_individual(trukan_learn_individual, dataset):
    model = trukan_learn_individual
    dataset = dataset

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "trukan_learn_individual"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()

    assert os.path.isfile("./figures/trukan_learn_individual.svg")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
