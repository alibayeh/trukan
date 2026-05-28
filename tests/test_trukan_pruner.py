import os

import pytest
import torch
from src.trukan.trukan import TruKan
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.trukan_pruner import TruKanPruner
from src.trukan.utils import create_dataset

from .utils import train


@pytest.fixture
def device():
    yield "cpu"


@pytest.fixture
def dataset():
    def f(x):
        return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    yield dataset


@pytest.fixture
def model(device):
    model = TruKan(
        layers_hidden=[2, 3, 1],
        num_knots=3,
        degree=3,
        knots_range=(-2, 2),
        learn_knots=False,
        device=device,
    )
    model = model.to(device)

    yield model


@pytest.fixture
def trained_model_meta(model, dataset):
    torch.set_default_dtype(torch.float64)
    model(dataset["train_input"])

    trained_model, model_metadata, _ = train(model, dataset, n_iteration=1000)
    trained_model.eval()
    model_metadata.compute_scores(dataset["train_input"])
    yield trained_model, model_metadata


def test_pruning(trained_model_meta, dataset, device):
    trained_model, model_metadata = trained_model_meta
    num_knots = trained_model.num_knots

    TruKanPlotter.setup_plot(
        model=trained_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_sub_components().with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "truKan_prun_test_1"),
    ).build_plot(model_metadata)

    pruner = TruKanPruner(trained_model)
    pruner.prune(dataset["test_input"], threshold=0.3, prune_type="neuron")
    smaller_model = pruner.create_structural_pruned_model()
    smaller_model = smaller_model.to(device)

    TruKanPlotter.setup_plot(
        model=smaller_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_sub_components().with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "truKan_prun_test_2"),
    ).build_plot()


def test_autopruning(trained_model_meta, dataset, device):
    trained_model, model_metadata = trained_model_meta
    num_knots = trained_model.num_knots

    TruKanPlotter.setup_plot(
        model=trained_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_sub_components().with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "truKan_autoprun_test_1"),
    ).build_plot(model_metadata)

    pruner = TruKanPruner(trained_model)
    pruner.auto_prune(dataset["test_input"], prune_type="neuron")
    smaller_model = pruner.create_structural_pruned_model()
    smaller_model = smaller_model.to(device)

    TruKanPlotter.setup_plot(
        model=smaller_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_sub_components().with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "truKan_autoprun_test_2"),
    ).build_plot()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
