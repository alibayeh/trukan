import os

import pytest
import torch
from src.trukan.trukan import TruKan
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.utils import create_dataset


@pytest.fixture
def device():
    yield "cpu"


@pytest.fixture
def dataset():
    def f(x):
        return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    yield dataset


@pytest.fixture(
    params=[
        {"name": "fixed_shared", "learn": False, "shared": True},
        {"name": "fixed_individual", "learn": False, "shared": False},
        {"name": "learn_shared", "learn": True, "shared": True},
        {"name": "learn_individual", "learn": True, "shared": False},
    ]
)
def model(request, device):
    config = request.param
    learn = config["learn"]
    shared = config["shared"]

    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=learn,
        shared_knots=shared,
        device=device,
    )
    model = model.to(device)

    yield model


def test_trukan_plot(model, dataset):
    learn = list(model.layers.values())[0].learn_knots
    share = list(model.layers.values())[0].shared_knots
    suffix = "_learn" if learn else "_fixed"
    suffix += "_share" if share else "_indiv"

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", f"trukan{suffix}"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()

    assert os.path.isfile(f"./figures/trukan{suffix}.svg")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
