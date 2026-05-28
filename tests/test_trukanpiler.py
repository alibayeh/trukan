import os

import pytest
import torch
from src.trukan.trukan_plotter import TruKanPlotter
from src.trukan.trukanpiler import trukanpiler, validate_compiled_model
from src.trukan.utils import create_dataset
from sympy import exp, sin, symbols


@pytest.fixture
def dataset():
    def f(x):
        # return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
        return torch.sin(torch.exp(x[:, 0]) + x[:, 1] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000, device="cpu")
    yield dataset


def test_trukanpiler_with_plot(dataset):
    x, y = symbols("x y")
    model = trukanpiler([x, y], sin(exp(x) + y**2))
    z = torch.rand(100, 2) * 2 - 1
    # _out ≈ sin(exp(z[:,0]) + z[:,1]**2)
    out = model(z)
    print(f"equation output: {torch.sin(torch.exp(z[:, 0]) + z[:, 1] ** 2)}")
    print(f"model output: {out}")
    TruKanPlotter.setup_plot(
        model=model,
        data=dataset["test_input"],
        num_knots=5,
    ).save_to(
        path=os.path.join("./figures", "trukanpiler"), format="svg"
    ).with_grid().with_sub_components().with_scatter_points().build_plot()


if __name__ == "__main__":
    x, y = symbols("x y")
    input_variables = [x, y]
    expression = sin(exp(x) + y**2)
    model = trukanpiler(input_variables, expression, num_knots=10)
    # z = torch.rand(100, 2) * 2 - 1
    # out = model(z)
    # f_out = torch.sin(torch.exp(z[:, 0]) + z[:, 1] ** 2)
    # print(f"equation output: {f_out}")
    # print(f"model output: {out.squeeze()}")

    # loss_fn = torch.nn.L1Loss()
    # absolute_error = loss_fn(f_out, out.squeeze())
    # print(f"absolute_error: {absolute_error.item()}")

    # loss_tensor = torch.abs(f_out - out.squeeze())
    # print(f"absolute_error: {torch.sum(loss_tensor).item()}")

    print(validate_compiled_model(input_variables, expression, model, n_test=200))
