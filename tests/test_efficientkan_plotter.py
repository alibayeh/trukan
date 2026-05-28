import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.utils import create_dataset

from .test_utils import train

############ # Copied from https://github.com/Blealtan/efficient-kan


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        name: str = "",  # <<<<< The name is needed by plotter to keep track of the layer's parameters
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def out_dim(self):  # <<<<< For metadata
        return self.out_features

    @property
    def coeffs(self):  # <<<<< For metadata
        return self.scaled_spline_weight()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_activated = self.base_activation(x)
        base_per_connection = base_activated[:, None, :] * self.base_weight[None, :, :]
        spline_basis = self.b_splines(x)
        num_splines = spline_basis.size(-1)
        spline_weight_reshaped = self.scaled_spline_weight.view(
            self.out_features, self.in_features, num_splines
        )
        spline_per_connection = torch.einsum(
            "bik,jik->bji", spline_basis, spline_weight_reshaped
        )
        intermediate = base_per_connection + spline_per_connection
        y = torch.sum(intermediate, dim=2)
        y = y.reshape(*original_shape[:-1], self.out_features)
        return y, intermediate.view((*original_shape, self.out_features)), None, None


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.layers_hidden = layers_hidden  # <<<<< For plotter
        self.grid_size = grid_size
        self.spline_order = spline_order

        # self.layers = torch.nn.ModuleList()
        self.layers: dict[str, KANLinear] = torch.nn.ModuleDict()  # <<<<< For plotter
        # for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden, layers_hidden[1:])
        ):
            name = f"KANLinear{i}"  # <<<<< Name is needed to keep track of layer's parameters
            self.layers[name] = KANLinear(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                name=name,  # <<<<< For plotter
            )

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
    model = KAN(
        layers_hidden=[2, 3, 1],
        grid_size=5,
        spline_order=3,
        grid_range=[-2, 2],
        scale_spline=1.0,
        enable_standalone_scale_spline=False,
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
        path=os.path.join("./figures", "efficientKan_saved"), format="svg"
    ).build_plot()

    assert os.path.isfile("./figures/efficientKan_saved.svg")


def test_metadata(setup_kan, setup_dataset):
    model = setup_kan
    dataset = setup_dataset

    num_knots = model.grid_size
    torch.set_default_dtype(torch.float64)

    device = "cpu"
    model = model.to(device)
    model(dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "efficientKan_prun_test_1"),
    ).build_plot()

    trained_model, model_attr, _ = train(model, dataset, n_iteration=1000)
    trained_model.eval()
    model_attr.compute_scores(dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=trained_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "efficientKan_prun_test_2"),
    ).build_plot(model_attr)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
