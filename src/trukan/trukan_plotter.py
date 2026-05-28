from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d

from .trukan_metadata import TruKanMetadata
from .utils import draw_neuron


def register_hooks(model, layer_class) -> tuple[dict, dict]:
    intermediates_in = {}
    intermediates_out = {}

    handles = []
    for module in model.modules():
        if isinstance(module, layer_class):

            def update_intermediates(mod, inp, out):
                intermediates_in.update(
                    {
                        mod.name: inp[0].detach().cpu().numpy()
                        if isinstance(inp, tuple)
                        else inp.detach().cpu().numpy()
                    }
                )
                intermediates_out.update({mod.name: out.detach().cpu().numpy()})

            hook = module.register_forward_hook(update_intermediates)
            handles.append(hook)

    return intermediates_in, intermediates_out


class SetupPlot:
    def __init__(
        self, model: torch.nn.Module, data, num_knots: int | None, scale, beta
    ) -> None:
        if not hasattr(model, "forward_traced") or not hasattr(model, "layers_hidden"):
            raise ValueError(
                "Must provide a model with forward_traced() method and layers_hidden property."
            )

        self._model = model
        self._data = data
        self._num_knots = num_knots
        self._scale = scale
        self._beta = beta
        self._draw_grid = False
        self._sub_components = False
        self._scatter_points = False
        self._full_path = None
        self._save_format = None
        self._colors = dict(
            in_out="black",
            connections="black",
            activations="darkslategray",
            main_curve="black",
            poly_curve="orange",
            trunc_curve="dodgerblue",
            grid="gray",
            scatter_point="red",
            sum_symbol="black",
        )

    def with_grid(self, color: str | None = None) -> SetupPlot:
        self._draw_grid = True
        if color is not None:
            self._colors["grid"] = color
        return self

    def with_sub_components(
        self,
        poly_curve_color: str | None = None,
        trunc_curve_color: str | None = None,
    ) -> SetupPlot:
        self._sub_components = True
        if poly_curve_color is not None:
            self._colors["poly_curve"] = poly_curve_color
        if trunc_curve_color is not None:
            self._colors["trunc_curve"] = trunc_curve_color
        return self

    def with_scatter_points(self, color: str | None = None) -> SetupPlot:
        self._scatter_points = True
        if color is not None:
            self._colors["scatter_point"] = color

        return self

    def save_to(self, path: str, format: str = "svg") -> SetupPlot:
        # if not os.path.exists(path):
        # os.makedirs(path)

        full_path = path if path[-3:] == format else f"{path}.{format}"
        self._full_path = full_path
        self._save_format = format
        return self

    def _get_normalized_figure_coordinates(self, fig: Figure, ax: Axes, x):
        return fig.transFigure.inverted().transform(ax.transData.transform(x))

    def build_plot(self, truKan_attribution: TruKanMetadata | None = None):
        # intermediates_in, intermediates_out = register_hooks(model)
        matplotlib.use("Agg")
        inputs, intermediates, intermediate_d, intermediate_n = (
            self._model.forward_traced(self._data)
        )

        layers_hidden = np.array(self._model.layers_hidden)
        y0 = 0.3  # height: from input to pre-mult
        z0 = 0.04  # height: from pre-mult to post-mult (input of next layer)

        neuron_depth = len(layers_hidden)
        min_spacing = 1 / np.maximum(np.max(layers_hidden), 5)

        max_neuron = np.max(layers_hidden)
        max_num_weights = np.max(layers_hidden[:-1] * layers_hidden[1:])
        # size (height/width) of 1D function diagrams
        y1 = 0.47 / np.maximum(max_num_weights, 5)
        # size (height/width) of operations (sum and mult)
        y2 = 0.15 / np.maximum(max_neuron, 5)

        fig, ax = plt.subplots(
            figsize=(
                10 * self._scale,
                10 * self._scale * (neuron_depth - 1) * (y0 + z0),
            ),
            dpi=600,
        )

        def score2alpha(_score):
            return np.tanh(self._beta * _score)

        if (
            truKan_attribution is not None
            and truKan_attribution.edge_scores is not None
        ):
            alpha = [
                score2alpha(score.cpu().detach().numpy())
                for score in truKan_attribution.edge_scores
            ]
        else:
            alpha = []

        # plot scatters and lines
        for depth in range(neuron_depth):
            n_act = layers_hidden[depth]

            # scatters inputs/outputs
            if depth == 0 or depth == len(layers_hidden) - 1:
                for i in range(n_act):
                    plt.scatter(
                        1 / (2 * n_act) + i / n_act,
                        depth * (y0 + z0),
                        s=min_spacing**2 * 5000 * self._scale**2,
                        color=self._colors.get("in_out"),
                    )

            # plot connections
            for i in range(n_act):
                if depth < neuron_depth - 1:
                    n_next = layers_hidden[depth + 1]
                    N = n_act * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j
                        try:
                            plt.plot(
                                [1 / (2 * n_act) + i / n_act, 1 / (2 * N) + id_ / N],
                                [depth * y0, depth * y0 + y0 / 2 - y1],
                                color=self._colors.get("connections"),
                                lw=2 * self._scale,
                                alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                            )
                            plt.plot(
                                [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                [depth * y0 + y0 / 2 + y1, depth * y0 + y0],
                                color=self._colors.get("connections"),
                                lw=2 * self._scale,
                                alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                            )
                        except Exception as e:
                            print(
                                f"alpha: {[np.shape(k) for k in alpha]}, depth: {depth}, j: {j}, i: {i}, {e}"
                            )

            if depth == neuron_depth - 2:
                plt.plot(
                    [
                        1 / (2 * layers_hidden[depth + 1]),
                        1 / (2 * layers_hidden[depth + 1]),
                    ],
                    [depth * (y0 + z0) + y0 / 2, (depth + 1) * (y0 + z0)],
                    color=self._colors.get("connections"),
                    lw=2 * self._scale,
                )

            plt.xlim(0, 1)
            plt.ylim(-0.1 * (y0 + z0), (neuron_depth - 1 + 0.1) * (y0 + z0))

        plt.axis("off")

        for depth in range(neuron_depth - 1):
            # plot splines
            n = layers_hidden[depth]
            for i in range(n):
                n_next = layers_hidden[depth + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j

                    left = self._get_normalized_figure_coordinates(
                        fig, ax, [1 / (2 * N) + id_ / N - y1, 0]
                    )[0]
                    right = self._get_normalized_figure_coordinates(
                        fig, ax, [1 / (2 * N) + id_ / N + y1, 0]
                    )[0]
                    bottom = self._get_normalized_figure_coordinates(
                        fig, ax, [0, depth * y0 + y0 / 2 - y1]
                    )[1]
                    up = self._get_normalized_figure_coordinates(
                        fig, ax, [0, depth * y0 + y0 / 2 + y1]
                    )[1]
                    newax = fig.add_axes((left, bottom, right - left, up - bottom))

                    newax.spines[:].set_visible(False)

                    rounded_box = draw_neuron(
                        xy=(0.015, 0.015),
                        width=0.97,
                        height=0.97,  # Position and size in Axes coordinates
                        radius=0.08,
                        linewidth=1.4,
                        edgecolor=self._colors.get("activations"),
                        facecolor="none",
                        transform=newax.transAxes,
                        zorder=5,
                        alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                    )
                    newax.add_patch(rounded_box)

                    inset_margin = 0.06  # 6% margin from each side
                    inner_ax = inset_axes(
                        newax,
                        width=f"{100 - 2 * inset_margin * 100}%",  # e.g., 90%
                        height=f"{100 - 2 * inset_margin * 100}%",
                        loc="center",
                    )
                    inner_ax.spines[:].set_visible(False)

                    rank = np.argsort(inputs[depth][:, i])
                    x_values = inputs[depth][:, i][rank]
                    y_values = intermediates[depth][:, i, j][rank]
                    x_min = np.min(x_values)
                    x_max = np.max(x_values)
                    grid_positions = np.linspace(x_min, x_max, self._num_knots)

                    if (
                        self._sub_components
                        and len(intermediate_d) > 0
                        and len(intermediate_n) > 0
                    ):
                        inner_ax.plot(
                            x_values,
                            intermediate_d[depth][:, i, j][rank],
                            color=self._colors.get("poly_curve"),
                            linestyle="dashed",
                            linewidth=0.7,
                            alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                        )
                        inner_ax.plot(
                            x_values,
                            intermediate_n[depth][:, i, j][rank],
                            color=self._colors.get("trunc_curve"),
                            linestyle="dashdot",
                            linewidth=0.7,
                            markersize=1,
                            alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                        )
                        y_min = np.min(
                            np.concat(
                                (
                                    y_values,
                                    intermediate_d[depth][:, i, j][rank],
                                    intermediate_n[depth][:, i, j][rank],
                                ),
                                axis=0,
                            )
                        )
                        y_max = np.max(
                            np.concat(
                                (
                                    y_values,
                                    intermediate_d[depth][:, i, j][rank],
                                    intermediate_n[depth][:, i, j][rank],
                                ),
                                axis=0,
                            )
                        )
                    else:
                        y_min = np.min(y_values)
                        y_max = np.max(y_values)

                    if self._draw_grid:
                        inner_ax.vlines(
                            x=grid_positions,
                            ymin=y_min + y_min * 0.15,
                            ymax=y_max + y_max * 0.15,
                            colors=self._colors.get("grid"),
                            linestyles="dotted",
                            linewidth=0.5,
                            alpha=alpha[depth][i][j]
                            if len(alpha) > 0 and alpha[depth][i][j] < 0.6
                            else 0.6,
                        )

                    inner_ax.plot(
                        x_values,
                        y_values,
                        color=self._colors.get("main_curve"),
                        linewidth=1.0,
                        alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                    )
                    if self._scatter_points:
                        interp_func = interp1d(
                            x_values,
                            y_values,
                            kind="linear",
                        )
                        y_at_intersections = interp_func(grid_positions)
                        inner_ax.scatter(
                            grid_positions,
                            y_at_intersections,
                            color=self._colors.get("scatter_point"),
                            s=1.1,
                            zorder=6,
                            alpha=alpha[depth][i][j] if len(alpha) > 0 else 1.0,
                        )
                    inner_ax.set_xticks([])
                    inner_ax.set_yticks([])

                    newax.set_xticks([])
                    newax.set_yticks([])
                    newax.set_facecolor("white")

            # plot sum symbols
            N = n = layers_hidden[depth + 1]
            for j in range(n):
                id_ = j
                left = self._get_normalized_figure_coordinates(
                    fig, ax, [1 / (2 * N) + id_ / N - y2, 0]
                )[0]
                right = self._get_normalized_figure_coordinates(
                    fig, ax, [1 / (2 * N) + id_ / N + y2, 0]
                )[0]
                bottom = self._get_normalized_figure_coordinates(
                    fig, ax, [0, depth * y0 + y0 - y2]
                )[1]
                up = self._get_normalized_figure_coordinates(
                    fig, ax, [0, depth * y0 + y0 + y2]
                )[1]
                newax = fig.add_axes((left, bottom, right - left, up - bottom))

                if depth == len(alpha) - 1:
                    symbol_alpha = 1.0
                else:
                    symbol_alpha = (
                        np.mean(alpha[depth], axis=0)[j] if len(alpha) > 0 else 1.0
                    )
                circle = Circle(
                    (0.5, 0.5),
                    0.4,
                    facecolor="white",
                    edgecolor=self._colors.get("sum_symbol"),
                    linewidth=1.5,
                    alpha=symbol_alpha,
                )
                newax.add_patch(circle)

                newax.text(
                    0.5,
                    0.5,
                    "+",
                    fontsize=20 * self._scale,
                    ha="center",
                    va="center",
                    weight="bold",
                    color=self._colors.get("sum_symbol"),
                    alpha=symbol_alpha,
                )
                newax.axis("off")

        if self._full_path is not None:
            fig.savefig(self._full_path, bbox_inches="tight", format=self._save_format)


class TruKanPlotter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def setup_plot(
        model,
        data,
        num_knots: int | None = None,
        scale=0.5,
        beta=3.0,
    ) -> SetupPlot:
        return SetupPlot(model, data, num_knots, scale, beta)
