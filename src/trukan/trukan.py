import random

import numpy as np
import torch
import torch.nn as nn

from .trukan_layer import TruKanLayer


class TruKan(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        num_knots=5,
        degree=3,
        # knots_init=None,
        learn_knots=False,
        shared_knots=True,
        knots_range=(-1, 1),
        eps=1e-8,
        seed=1,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.num_knots = num_knots
        self.degree = degree
        self.knots_range = knots_range
        self.layers_hidden = layers_hidden

        self.layers: dict[str, TruKanLayer] = nn.ModuleDict()
        for i, (in_features, out_features) in enumerate(
            zip(layers_hidden, layers_hidden[1:])
        ):
            name = f"TruKanLayer_{i}"
            self.layers[name] = TruKanLayer(
                in_dim=in_features,
                out_dim=out_features,
                num_knots=self.num_knots,
                degree=self.degree,
                learn_knots=learn_knots,
                shared_knots=shared_knots,
                knots_range=self.knots_range,
                eps=eps,
                device=device,
                name=name,
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        inputs = []
        intermediates = []
        intermediates_poly = []
        intermediates_trunc = []
        for layer in self.layers.values():
            inputs.append(x.detach().cpu().numpy())
            x, intermediate, intermediate_d, intermediate_n = layer.forward_traced(x)
            intermediate_np = intermediate.detach().cpu().numpy()
            intermediates.append(intermediate_np)

            intermediates_poly.append(intermediate_d.detach().cpu().numpy())
            intermediates_trunc.append(intermediate_n.detach().cpu().numpy())
        return inputs, intermediates, intermediates_poly, intermediates_trunc
