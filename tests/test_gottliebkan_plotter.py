import os
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.utils import create_dataset

from .utils import train

############ # Copied from https://github.com/seydi1370/Basis_Functions/blob/main/Gottlieb.py


def gottlieb(n, x, alpha):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * alpha * x
    else:
        return 2 * (alpha + n - 1) * x * gottlieb(n - 1, x, alpha) - (
            alpha + 2 * n - 2
        ) * gottlieb(n - 2, x, alpha)


class GottliebKANLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        degree,
        name: str = "",  # <<<<< The name is needed by plotter to keep track of the layer's parameters
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.alpha = nn.Parameter(torch.randn(1))

        self.gottlieb_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.gottlieb_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    @property
    def out_dim(self):  # <<<<< For metadata
        return self.output_dim

    @property
    def coeffs(self):  # <<<<< For metadata
        return self.gottlieb_coeffs

    def forward(self, x):
        x = torch.sigmoid(x)
        gottlieb_basis = []
        for n in range(self.degree + 1):
            gottlieb_basis.append(gottlieb(n, x, self.alpha))
        gottlieb_basis = torch.stack(gottlieb_basis, dim=-1)
        y = torch.einsum("bid,iod->bo", gottlieb_basis, self.gottlieb_coeffs)
        y = y.view(-1, self.output_dim)
        return y

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        original_shape = x.shape
        x = x.reshape(-1, self.input_dim)
        x = torch.sigmoid(x)
        gottlieb_basis = []
        for n in range(self.degree + 1):
            gottlieb_basis.append(gottlieb(n, x, self.alpha))
        gottlieb_basis = torch.stack(gottlieb_basis, dim=-1)
        gottlieb_per_connection = torch.einsum(
            "bid,iod->bio", gottlieb_basis, self.gottlieb_coeffs
        )
        y = gottlieb_per_connection.sum(dim=1)
        y = y.reshape(*original_shape[:-1], self.output_dim)
        intermediate = gottlieb_per_connection.reshape(
            *original_shape[:-1], self.input_dim, self.output_dim
        )
        return y, intermediate, None, None


class GottliebKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=3,
    ):
        super().__init__()
        self.layers_hidden = layers_hidden  # <<<<< For plotter
        self.degree = degree

        # self.layers = torch.nn.ModuleList()
        self.layers: dict[str, GottliebKANLayer] = (
            torch.nn.ModuleDict()
        )  # <<<<< For plotter
        # for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden, layers_hidden[1:])
        ):
            name = f"KANLayer{i}"  # <<<<< Name is needed to keep track of layer's parameters
            self.layers[name] = GottliebKANLayer(
                in_features,
                out_features,
                degree=degree,
                name=name,  # <<<<< For plotter
            )

    @property
    def grid_size(self):  # <<<<< For metadata
        return self.degree + 1

    def forward(self, x: torch.Tensor):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        inputs = []
        intermediates = []
        for layer in self.layers.values():
            inputs.append(x.detach().cpu().numpy())
            x, intermediate, _, _ = layer.forward_traced(x)
            intermediate_np = intermediate.detach().cpu().numpy()
            intermediates.append(intermediate_np)

        return inputs, intermediates, [], []


##########################################################################################################


@pytest.fixture
def setup_dataset():
    def f(x):
        return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    yield dataset


@pytest.fixture
def setup_kan():
    model = GottliebKAN(
        layers_hidden=[2, 3, 1],
        degree=3,
    )
    model = model.to("cpu")

    yield model


def test_plotting(setup_kan, setup_dataset):
    model = setup_kan
    dataset = setup_dataset

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "gottliebkan_saved"), format="svg"
    ).build_plot()

    assert os.path.isfile("./figures/gottliebkan_saved.svg")


def test_metadata(setup_kan, setup_dataset):
    model = setup_kan
    dataset = setup_dataset

    num_knots = (
        model.grid_size() if isinstance(model.grid_size, Callable) else model.grid_size
    )
    torch.set_default_dtype(torch.float64)

    device = "cpu"
    model = model.to(device)
    model(dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "gottliebkan_prun_test_1"),
    ).build_plot()

    trained_model, model_attr, _ = train(model, dataset, n_iteration=1000)
    trained_model.eval()
    model_attr.compute_scores(dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=trained_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "gottliebkan_prun_test_2"),
    ).build_plot(model_attr)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
