import os
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.utils import create_dataset

from .test_utils import train

############ # Copied from https://github.com/SynodicMonth/ChebyKAN


# This is inspired by Kolmogorov-Arnold Networks
# but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        degree,
        name: str = "",  # <<<<< The name is needed by plotter to keep track of the layer's parameters
    ):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    @property
    def out_dim(self):  # <<<<< For metadata
        return self.outdim

    @property
    def coeffs(self):  # <<<<< For metadata
        return self.cheby_coeffs

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        original_shape = x.shape
        x = x.reshape(-1, self.inputdim)  # (batch, in)

        # Normalize input to [-1, 1] domain of Chebyshev polynomials
        x_tanh = torch.tanh(x)  # (batch, in)

        # Expand for each polynomial degree: (batch, in, degree+1)
        x_expanded = x_tanh.unsqueeze(-1).expand(-1, -1, self.degree + 1)

        # T_k(x) = cos(k * arccos(x))
        cheby_basis = torch.cos(self.arange * torch.acos(x_expanded))
        # shape: (batch, inputdim, degree+1)

        # --- Per-connection contributions ---
        # cheby_coeffs: (inputdim, outdim, degree+1)
        # Sum over degree axis only → keep (batch, out, in) per-connection view
        # einsum: b=batch, i=in, d=degree, o=out  →  "bid,iod->boi"
        cheby_per_connection = torch.einsum(
            "bid,iod->boi", cheby_basis, self.cheby_coeffs
        )  # (batch, outdim, inputdim)

        # Sum over in_features to produce final output
        y = cheby_per_connection.sum(dim=2)  # (batch, outdim)
        y = y.reshape(*original_shape[:-1], self.outdim)
        # Reshape intermediate to (*original_shape, outdim) to match efficientKAN convention
        # intermediate = cheby_per_connection.view(*original_shape, self.outdim)
        intermediate = cheby_per_connection.reshape(*original_shape, self.outdim)
        # No base branch, no separate spline scaler → both extra slots are None
        return y, intermediate, None, None


class ChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=3,
    ):
        super().__init__()
        self.layers_hidden = layers_hidden  # <<<<< For plotter
        self.degree = degree

        # self.layers = torch.nn.ModuleList()
        self.layers: dict[str, ChebyKANLayer] = (
            torch.nn.ModuleDict()
        )  # <<<<< For plotter
        # for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden, layers_hidden[1:])
        ):
            name = f"KANLinear{i}"  # <<<<< Name is needed to keep track of layer's parameters
            self.layers[name] = ChebyKANLayer(
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
    model = ChebyKAN(
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
        path=os.path.join("./figures", "chebyKan_saved"), format="svg"
    ).build_plot()

    assert os.path.isfile("./figures/chebyKan_saved.svg")


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
        os.path.join("./figures", "chebyKan_prun_test_1"),
    ).build_plot()

    trained_model, model_attr, _ = train(model, dataset, n_iteration=1000)
    trained_model.eval()
    model_attr.compute_scores(dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=trained_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "chebyKan_prun_test_2"),
    ).build_plot(model_attr)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
