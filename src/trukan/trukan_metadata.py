from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class TruKanMetadata:
    def __init__(self, model):
        self._model = model
        self.xy_sample = {}
        self.edge_actscale = []  # [layer][in_dim, out_dim]
        self.node_actscale = []  # [layer][out_dim]
        self.intermediate_actscale = []  # [layer][in_dim, out_dim]

        # Storage for scores
        self.node_scores_all = None
        self.edge_scores_all = None
        self.node_scores = None
        self.edge_scores = None

    def compute_activation_scales(self, x: torch.Tensor, n_samples: int = 100):
        if x.shape[0] > n_samples:
            indices = torch.randperm(x.shape[0])[:n_samples]
            x = x[indices]

        self.edge_actscale = []
        self.node_actscale = []
        self.intermediate_actscale = []

        # Run forward pass and collect activations
        current_input = x

        for layer_name, layer in self._model.layers.items():
            with torch.no_grad():
                output, intermediate, intermediate_d, intermediate_n = (
                    layer.forward_traced(current_input)
                )
                if self.xy_sample.get(layer_name) is None:
                    self.xy_sample[layer_name] = [current_input, intermediate]

                edge_scale = torch.std(intermediate, dim=0)  # [in_dim, out_dim]
                self.edge_actscale.append(edge_scale)

                intermediate_scale = torch.std(intermediate, dim=0)  # [in_dim, out_dim]
                self.intermediate_actscale.append(intermediate_scale)

                node_scale = torch.std(output, dim=0)  # [out_dim]
                self.node_actscale.append(node_scale)

                current_input = output

        return self

    def compute_scores(
        self,
        x: torch.Tensor | None = None,
        layer_idx: int | None = None,
        out_score: torch.Tensor | None = None,
        n_samples: int = 100,
    ) -> TruKanMetadata:
        # Compute activation scales if not already done
        if len(self.edge_actscale) == 0:
            if x is None:
                raise ValueError("Must provide input data x for first call")
            self.compute_activation_scales(x, n_samples)

        # Determine which layer to start from
        num_layers = len(self._model.layers)
        start_layer = num_layers if layer_idx is None else layer_idx

        # Get output dimension
        layer_list = list(self._model.layers.values())
        out_dim = layer_list[start_layer - 1].out_dim
        out_dim = out_dim if start_layer > 0 else self._model.layers_hidden[-1]

        # Initialize output importance scores
        device = next(self._model.parameters()).device
        if out_score is None:
            node_score = torch.eye(out_dim, device=device)
        else:
            if out_score.dim() == 1:
                node_score = torch.diag(out_score.to(device))
            else:
                node_score = out_score.to(device)

        node_scores = [node_score]
        edge_scores = []

        # Backpropagate through layers
        for l_idx in range(start_layer - 1, -1, -1):
            edge_score = self._backprop_layer(l_idx, node_score)
            edge_scores.append(edge_score)
            node_score = edge_score.sum(
                dim=-1
            )  # [out_dim_curr, in_dim] -> sum over in_dim contributions
            node_scores.append(node_score)

        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))

        self.node_scores = [
            scores.mean(dim=0) if scores.dim() > 1 else scores
            for scores in self.node_scores_all
        ]
        self.edge_scores = [
            scores.mean(dim=0) if scores.dim() > 2 else scores
            for scores in self.edge_scores_all
        ]
        return self

    def get_scores(
        self,
        layer_idx: int | None = None,
        neuron_idx: int | None = None,
    ) -> torch.Tensor:
        if layer_idx is not None and neuron_idx is not None:
            return self._return_neuron_scores(neuron_idx)

        else:
            if self.node_scores_all is None:
                raise ValueError("Must call compute_scores() first")

            return self.node_scores_all[0]

    def _backprop_layer(self, layer_idx: int, node_score: torch.Tensor) -> torch.Tensor:
        edge_scale = self.edge_actscale[layer_idx]  # [in_dim, out_dim]
        node_scale = self.node_actscale[layer_idx]  # [out_dim]

        # Edge contribution = edge_scale * node_importance / node_scale
        # [out_dim_curr, out_dim]
        normalized_node_score = node_score / (node_scale.unsqueeze(0) + 1e-8)

        # edge_score[k, i, j] = edge_scale[i,j] * normalized_node_score[k,j]
        edge_score = edge_scale.unsqueeze(0) * normalized_node_score.unsqueeze(1)
        return edge_score

    def _return_neuron_scores(self, neuron_idx: int) -> torch.Tensor:
        if self.node_scores_all is None:
            raise ValueError("Must call compute_scores() first")

        scores = self.node_scores_all[0][neuron_idx]

        return scores

    def plot_layer_scores(self, layer_idx: int | None = None):
        if self.node_scores is None:
            raise ValueError("Must call compute_scores() first")

        idx = 0 if layer_idx is None else layer_idx
        scores = self.node_scores[idx].cpu().detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(scores)), scores, alpha=0.7, edgecolor="black")
        plt.xlabel("Feature Index", fontsize=12)
        plt.ylabel("Mean Score", fontsize=12)
        plt.title(f"Feature Scores (Layer {idx})", fontsize=14, fontweight="bold")
        plt.xticks(range(len(scores)))
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_network_flow(self):
        if self.edge_scores_all is None:
            raise ValueError("Must call get_scores() first")

        num_layers = len(self.edge_scores_all)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))

        if num_layers == 1:
            axes = [axes]

        for idx, edge_scores in enumerate(self.edge_scores_all):
            if edge_scores.dim() == 3:
                # Average across output dimensions if needed
                edge_matrix = edge_scores.mean(dim=0)
            else:
                edge_matrix = edge_scores

            edge_np = edge_matrix.cpu().detach().numpy()

            sns.heatmap(
                edge_np,
                ax=axes[idx],
                cmap="RdYlGn",
                center=0,
                cbar_kws={"label": "Scores"},
                xticklabels=True,
                yticklabels=True,
            )
            axes[idx].set_title(
                f"Layer {idx} → {idx + 1}", fontsize=12, fontweight="bold"
            )
            axes[idx].set_xlabel(f"Output Neurons (Layer {idx + 1})", fontsize=10)
            axes[idx].set_ylabel(f"Input Features (Layer {idx})", fontsize=10)

        plt.tight_layout()
        plt.show()

    def get_feature_importance(self) -> np.ndarray:
        if self.node_scores is None:
            raise ValueError("Must call get_scores() first")

        return self.node_scores[0].cpu().detach().numpy()

    def get_top_features(self, k: int = 5) -> list[tuple[int, float]]:
        importance = self.get_feature_importance()
        top_indices = np.argsort(np.abs(importance))[-k:][::-1]
        return [(int(idx), float(importance[idx])) for idx in top_indices]
