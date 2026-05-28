import math
import os

import numpy as np
import pytest
import torch
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.utils import create_dataset

from .test_utils import train

############ # Copied from https://github.com/ereinha/SineKAN


def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size ** (-K) + C
    i_n1 = ratio * i_n
    return i_n1


class SineKANLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        device="cuda",
        grid_size=5,
        is_first=False,
        add_bias=True,
        norm_freq=True,
        name: str = "",  # <<<<< The name is needed by plotter to keep track of the layer's parameters
    ):
        super().__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A, self.K, self.C = (
            0.9724108095811765,
            0.9884401790754128,
            0.999449553483052,
        )

        self.grid_norm_factor = torch.arange(grid_size) + 1
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)

        if is_first:
            self.amplitudes = torch.nn.Parameter(
                torch.empty(output_dim, input_dim, 1).normal_(0, 0.4)
                / output_dim
                / self.grid_norm_factor
            )
        else:
            self.amplitudes = torch.nn.Parameter(
                torch.empty(output_dim, input_dim, 1).uniform_(-1, 1)
                / output_dim
                / self.grid_norm_factor
            )

        grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (
            grid_size + 1
        )
        self.input_phase = (
            torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        )
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(
                torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size)
                / (grid_size + 1) ** (1 - is_first)
            )
        else:
            self.freq = torch.nn.Parameter(
                torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size)
            )

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        # self.phase = torch.nn.Parameter(phase)
        self.register_buffer("phase", phase)

        if self.add_bias:
            self.bias = torch.nn.Parameter(torch.ones(1, output_dim) / output_dim)

    @property
    def out_dim(self):  # <<<<< For metadata
        return self.output_dim

    @property
    def coeffs(self):  # <<<<< For metadata
        return self.amplitudes

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum("ijkl,jkl->ij", s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        original_shape = x.shape
        x = x.reshape(-1, self.input_dim)  # (batch, input_dim)

        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)
        # (batch, 1, input_dim, 1) * (1, 1, 1, grid_size) + (1, 1, input_dim, grid_size)
        s = torch.sin(x_reshaped * self.freq + self.phase)
        # s: (batch, 1, input_dim, grid_size)

        # Squeeze the broadcast singleton dim so indices are explicit
        s = s.squeeze(1)  # (batch, input_dim, grid_size)  →  b, k, l

        # --- Per-connection contributions ---
        # amplitudes: (output_dim, input_dim, grid_size)  →  j, k, l
        # Contract over grid_size (l) only; keep input_dim (k) per-connection
        # einsum: b=batch, k=input_dim, l=grid_size, j=output_dim  →  "bkl,jkl->bjk"
        sine_per_connection = torch.einsum(
            "bkl,jkl->bjk", s, self.amplitudes
        )  # (batch, output_dim, input_dim)

        # Sum over input_dim to produce output — mirrors torch.sum(intermediate, dim=2)
        y = sine_per_connection.sum(dim=2)  # (batch, output_dim)

        # Bias is a global per-output offset, not per-connection; add to y only
        if self.add_bias:
            y = y + self.bias

        y = y.reshape(*original_shape[:-1], self.output_dim)
        # Reshape to (*original_shape, output_dim) to match efficientKAN convention
        # intermediate = sine_per_connection.view(*original_shape, self.output_dim)
        intermediate = sine_per_connection.reshape(*original_shape, self.output_dim)
        return y, intermediate, None, None


class SineKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        grid_size: int = 8,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.layers_hidden = layers_hidden  # <<<<< For plotter
        self.grid_size = grid_size  # <<<<< For plotter

        self.layers: dict[str, SineKANLayer] = (
            torch.nn.ModuleDict()
        )  # <<<<< For plotter
        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden, layers_hidden[1:])
        ):
            name = f"SineKANLayer{i}"  # <<<<< Name is needed to keep track of layer's parameters
            self.layers[name] = SineKANLayer(
                in_features,
                out_features,
                grid_size=grid_size,
                is_first=i == 0,
                name=name,  # <<<<< For plotter
            )
        # self.layers = torch.nn.ModuleList(
        #     [
        #         SineKANLayer(
        #             in_dim, out_dim, device, grid_size=grid_size, is_first=True
        #         )
        #         if i == 0
        #         else SineKANLayer(
        #             in_dim,
        #             out_dim,
        #             device,
        #             grid_size=grid_size,
        #         )
        #         for i, (in_dim, out_dim) in enumerate(
        #             zip(layers_hidden[:-1], layers_hidden[1:])
        #         )
        #     ]
        # )

    def forward(self, x):
        # for layer in self.layers:
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
    model = SineKAN(
        layers_hidden=[2, 3, 1],
        grid_size=5,
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
        path=os.path.join("./figures", "sineKan_saved"), format="svg"
    ).build_plot()

    assert os.path.isfile("./figures/sineKan_saved.svg")


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
        os.path.join("./figures", "sineKan_prun_test_1"),
    ).build_plot()

    trained_model, model_attr, _ = train(model, dataset, n_iteration=1000)
    trained_model.eval()
    model_attr.compute_scores(dataset["train_input"])

    TruKanPlotter.setup_plot(
        model=trained_model,
        data=dataset["test_input"],
        num_knots=num_knots,
    ).with_grid().with_scatter_points().save_to(
        os.path.join("./figures", "sineKan_prun_test_2"),
    ).build_plot(model_attr)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
