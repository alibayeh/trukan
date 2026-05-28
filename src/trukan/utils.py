import numpy as np
import torch
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from torch.types import Device


def create_dataset(
    f,
    n_var=2,
    f_mode="col",
    ranges=[-1, 1],
    train_num=1000,
    test_num=1000,
    normalize_input=False,
    normalize_label=False,
    device: Device = "cpu",
    seed=0,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = (
            torch.rand(
                train_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )
        test_input[:, i] = (
            torch.rand(
                test_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )

    if f_mode == "col":
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == "row":
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        raise ValueError(f"f_mode {f_mode} not recognized")

    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)

    def normalize(data, mean, std):
        return (data - mean) / std

    if normalize_input:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)

    if normalize_label:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {
        "train_input": train_input.to(device),
        "test_input": test_input.to(device),
        "train_label": train_label.to(device),
        "test_label": test_label.to(device),
    }

    return dataset


def draw_neuron(
    xy: tuple[float, float], width: float, height: float, radius: float, **kwargs
):
    x, y = xy
    kappa = 0.5522847498307933
    verts = [
        (x + radius, y),
        (x + width - radius, y),
        (x + width - radius + radius * kappa, y),
        (x + width, y + radius - radius * kappa),
        (x + width, y + radius),
        (x + width, y + height - radius),
        (x + width, y + height - radius + radius * kappa),
        (x + width - radius + radius * kappa, y + height),
        (x + width - radius, y + height),
        (x + radius, y + height),
        (x + radius - radius * kappa, y + height),
        (x, y + height - radius + radius * kappa),
        (x, y + height - radius),
        (x, y + radius),
        (x, y + radius - radius * kappa),
        (x + radius - radius * kappa, y),
        (x + radius, y),
        (x + radius, y),
    ]

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]

    path = Path(verts, codes)
    return PathPatch(path, **kwargs)
