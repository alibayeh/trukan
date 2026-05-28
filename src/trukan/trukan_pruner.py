from __future__ import annotations

import torch

from .trukan import TruKan
from .trukan_metadata import TruKanMetadata


class TruKanPruner:
    def __init__(self, model, metadata: TruKanMetadata | None = None):
        self.model = model
        self.metadata = metadata if metadata is not None else TruKanMetadata(model)
        self.pruning_history = []
        self.masks = {}  # Store pruning masks for each layer

    def prune(
        self,
        x: torch.Tensor,
        threshold: float = 1e-2,
        mode: str = "auto",
        prune_type: str = "edge",
        n_samples: int = 100,
    ) -> dict:
        if self.metadata.edge_scores is None:
            print("Computing metadata...")
            self.metadata.compute_scores(x, n_samples=n_samples)

        if prune_type == "edge":
            stats = self._prune_edges(threshold, mode)
        elif prune_type == "neuron":
            stats = self._prune_neurons(threshold, mode)
        else:
            raise ValueError(
                f"Unknown prune_type: {prune_type}. Use 'edge' or 'neuron'"
            )

        self.pruning_history.append(stats)
        return stats

    def _prune(self, threshold: float, mode: str, prune_type: str) -> dict:
        total_key = f"total_{'edges' if prune_type == 'edge' else 'neurons'}"
        stats = {
            "prune_type": prune_type,
            "threshold": threshold,
            "mode": mode,
            "layers": [],
            "total_pruned": 0,
            total_key: 0,
        }

        n_layers = len(self.model.layers)

        for layer_idx, (layer_name, layer) in enumerate(self.model.layers.items()):
            # Neurons are scored at each layer's output, so the final layer
            # (which has no outgoing neurons to prune) is always skipped.
            if prune_type == "neuron" and layer_idx == n_layers - 1:
                continue

            edge_scores = self.metadata.edge_scores[layer_idx]  # [in_dim, out_dim]

            if prune_type == "edge":
                scores = edge_scores.abs()  # [in_dim, out_dim]
            else:  # neuron
                scores = edge_scores.sum(dim=0).abs()  # [out_dim]

            if mode == "auto":
                actual_threshold = threshold * scores.max()
            elif mode == "absolute":
                actual_threshold = threshold
            elif mode == "percentage":
                sorted_scores = torch.sort(scores.flatten())[0]
                percentile_idx = int(len(sorted_scores) * threshold / 100)
                actual_threshold = sorted_scores[percentile_idx]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # --- Build mask and zero out pruned coefficients ---
            mask = scores > actual_threshold
            self.masks[layer_name] = mask

            with torch.no_grad():
                if prune_type == "edge":
                    # mask: [in_dim, out_dim]  →  broadcast over coeff's last dim
                    expanded_mask = mask.unsqueeze(-1).expand_as(layer.coeffs)
                else:
                    # mask: [out_dim]  →  broadcast over in_dim and coeff's last dim
                    expanded_mask = (
                        mask.unsqueeze(0).unsqueeze(-1).expand_as(layer.coeffs)
                    )

                layer.coeffs.data *= expanded_mask.to(layer.coeffs.device)

            # --- Accumulate stats ---
            n_pruned = (~mask).sum().item()
            n_total = mask.numel()

            layer_stats = {
                "layer": layer_name,
                "pruned": n_pruned,
                "total": n_total,
                "kept": n_total - n_pruned,
                "prune_ratio": n_pruned / n_total if n_total > 0 else 0,
            }
            stats["layers"].append(layer_stats)
            stats["total_pruned"] += n_pruned
            stats[total_key] += n_total

        total_count = stats[total_key]
        stats["total_prune_ratio"] = (
            stats["total_pruned"] / total_count if total_count > 0 else 0
        )

        label = "Edge" if prune_type == "edge" else "Neuron"
        unit = "edges" if prune_type == "edge" else "neurons"
        print(f"\n{label} Pruning Complete:")
        print(
            f"Total {unit} pruned: {stats['total_pruned']} / {total_count}"
            f" ({stats['total_prune_ratio']:.2%})"
        )
        for ls in stats["layers"]:
            print(
                f"{ls['layer']}: {ls['pruned']} / {ls['total']} ({ls['prune_ratio']:.2%})"
            )

        return stats

    def _prune_edges(self, threshold: float, mode: str) -> dict:
        return self._prune(threshold, mode, prune_type="edge")

    def _prune_neurons(self, threshold: float, mode: str) -> dict:
        return self._prune(threshold, mode, prune_type="neuron")

    def auto_prune(
        self,
        x: torch.Tensor,
        target_sparsity: float = 0.5,
        prune_type: str = "edge",
        max_iterations: int = 10,
        n_samples: int = 100,
    ) -> list[dict]:
        print(f"\nAuto-pruning to {target_sparsity:.1%} sparsity...")

        all_stats = []
        current_sparsity = 0.0

        for iteration in range(max_iterations):
            self.metadata.compute_scores(x, n_samples=n_samples)

            # Adjust threshold to reach target
            remaining = target_sparsity - current_sparsity
            if remaining <= 0:
                break

            threshold = min(remaining * 100, 20)  # Prune max 20% per iteration

            stats = self.prune(
                x,
                threshold=threshold,
                mode="percentage",
                prune_type=prune_type,
                n_samples=n_samples,
            )

            all_stats.append(stats)
            current_sparsity += stats["total_prune_ratio"] * (1 - current_sparsity)

            print(f"  Iteration {iteration + 1}: Sparsity = {current_sparsity:.2%}")

            if current_sparsity >= target_sparsity * 0.95:  # Within 5% of target
                break

        print(f"\nFinal sparsity: {current_sparsity:.2%}")
        return all_stats

    def get_model_sparsity(self) -> dict[str, float]:
        total_params = 0
        zero_params = 0

        for _, layer in self.model.layers.items():
            params = layer.coeffs.numel()
            zeros = (layer.coeffs.abs() < 1e-8).sum().item()

            total_params += params
            zero_params += zeros

        sparsity = zero_params / total_params if total_params > 0 else 0

        return {
            "total_params": total_params,
            "zero_params": zero_params,
            "nonzero_params": total_params - zero_params,
            "sparsity": sparsity,
        }

    def get_pruned_model(self) -> TruKan:
        return self.model

    def create_structural_pruned_model(self, min_neurons: int = 1) -> TruKan:

        if not self.masks:
            raise ValueError("Must prune first before creating structural model")

        # Check if have neuron masks
        has_neuron_masks = any(mask.ndim == 1 for mask in self.masks.values())
        if not has_neuron_masks:
            print(
                "Warning: No neuron-level pruning detected. Using edge-based pruning to infer neuron removal."
            )
            return self._create_structural_from_edges(min_neurons)

        return self._create_structural_from_neurons(min_neurons)

    def _create_structural_from_neurons(self, min_neurons: int) -> TruKan:
        new_architecture = [self.model.layers_hidden[0]]  # Input dimension
        kept_indices = []  # Track which neurons to keep in each layer

        for layer_idx, (layer_name, layer) in enumerate(self.model.layers.items()):
            if layer_name in self.masks:
                mask = self.masks[layer_name]
                if mask.ndim == 1:  # Neuron mask
                    kept = torch.where(mask)[0]
                    n_kept = max(len(kept), min_neurons)
                    if len(kept) < min_neurons:
                        # Keep top min_neurons by score
                        neuron_scores = self.metadata.node_scores[layer_idx + 1]
                        top_indices = torch.argsort(
                            neuron_scores.abs(), descending=True
                        )[:min_neurons]
                        kept = top_indices
                    new_architecture.append(n_kept)
                    kept_indices.append(kept)
                else:
                    # No neuron pruning for this layer
                    new_architecture.append(layer.out_dim)
                    kept_indices.append(torch.arange(layer.out_dim))
            else:
                new_architecture.append(layer.out_dim)
                kept_indices.append(torch.arange(layer.out_dim))

        new_model = TruKan(
            layers_hidden=new_architecture,
            num_knots=self.model.num_knots,
            degree=self.model.degree,
            learn_knots=getattr(
                self.model.layers[list(self.model.layers.keys())[0]],
                "learn_knots",
                False,
            ),
            knots_range=self.model.knots_range,
            device=next(self.model.parameters()).device,
        )

        self._transfer_weights(new_model, kept_indices)
        return new_model

    def _create_structural_from_edges(self, min_neurons: int) -> TruKan:
        new_architecture = [self.model.layers_hidden[0]]
        kept_indices = []

        for layer_idx, (layer_name, layer) in enumerate(self.model.layers.items()):
            if layer_name in self.masks:
                mask = self.masks[layer_name]  # [in_dim, out_dim]
                active_per_neuron = mask.sum(dim=0) > 0  # [out_dim]
                kept = torch.where(active_per_neuron)[0]
                n_kept = max(len(kept), min_neurons)
                if len(kept) < min_neurons and layer_idx < len(self.model.layers) - 1:
                    neuron_importance = (
                        self.metadata.edge_scores[layer_idx].sum(dim=0).abs()
                    )
                    top_indices = torch.argsort(neuron_importance, descending=True)[
                        :min_neurons
                    ]
                    kept = top_indices

                new_architecture.append(n_kept)
                kept_indices.append(kept)
            else:
                new_architecture.append(layer.out_dim)
                kept_indices.append(torch.arange(layer.out_dim))

        new_model = TruKan(
            layers_hidden=new_architecture,
            num_knots=self.model.num_knots,
            degree=self.model.degree,
            knots_range=self.model.knots_range,
            device=next(self.model.parameters()).device,
        )

        self._transfer_weights(new_model, kept_indices)
        return new_model

    def _transfer_weights(self, new_model, kept_indices):
        old_layers = list(self.model.layers.values())
        new_layers = list(new_model.layers.values())

        for layer_idx, (new_layer, old_layer, kept_out) in enumerate(
            zip(new_layers, old_layers, kept_indices)
        ):
            with torch.no_grad():
                if layer_idx == 0:
                    kept_in = torch.arange(old_layer.in_dim)
                else:
                    kept_in = kept_indices[layer_idx - 1]

                old_coeffs = old_layer.coeffs[kept_in][:, kept_out, :]
                new_layer.coeffs.data = old_coeffs.clone()

                if hasattr(old_layer, "knots") and old_layer.knots is not None:
                    if old_layer.knots.requires_grad:
                        old_knots = old_layer.knots[kept_in][:, kept_out, :]
                        new_layer.knots.data = old_knots.clone()
