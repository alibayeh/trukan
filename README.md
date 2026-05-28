# trukan

[![arXiv](https://img.shields.io/badge/arXiv-2602.03879-b31b1b.svg)](https://arxiv.org/abs/2602.03879)

**TruKAN** is a Python package implementing a novel, efficient variation of Kolmogorov-Arnold Networks (KANs). It replaces the B-spline basis with truncated power functions derived from k-order spline theory, better interpretability, and strong performance on real-world tasks.

## About TruKAN

TruKAN preserves the canonical KAN topology and learnable univariate activations but replaces the computationally heavy B-spline basis with a family of **truncated power functions** `(x - t_j)_+^k` combined with a low-order polynomial term. This design maintains full expressiveness while improved the performance.

Key advantages demonstrated in the paper:
- **Efficiency**: Lower GPU memory compared to standard KAN (pykan) and some other variants.
- **Interpretability**: Explicit decomposition into polynomial + truncated-power terms makes learned functions easy to analyze and visualize.
- **Flexibility**: Supports both shared knots (more efficient) and individual knots (more expressive).
- **Performance**: Outperforms MLP, standard KAN, and SineKAN on CIFAR-10/100, STL-10, and Oxford-IIIT Pets when integrated into an EfficientNet-V2 backbone.

**Paper**: [TruKAN: Towards More Efficient Kolmogorov-Arnold Networks Using Truncated Power Functions](https://arxiv.org/abs/2602.03879)  
**Authors**: Ali Bayeh, Samira Sadaoui, Malek Mouhoub  
**arXiv**: 2602.03879 (February 2026)



**TruKAN Layer Structure**  
<table align="center">
  <tr>
    <td align="center">
      <img src="./assets/trukan_fixed.svg" height="200"><br>
      <sub>(a) Fixed-knot version (equal intervals)</sub>
    </td>
    <td align="center">
      <img src="./assets/trukan_learnable.svg" height="200"><br>
      <sub>(b) Learnable-knot version (trainable knot positions with ordering constraints)</sub>
    </td>
  </tr>
</table>

**Learned Activations**  
<table align="center">
  <tr>
    <td align="center">
      <img src="./assets/trukan_trained.svg" height="200"><br>
      <sub>(a) Trained TruKAN</sub>
    </td>
    <td align="center">
      <img src="./assets/trukan_pruned.svg" height="200"><br>
      <sub>(b) Pruned TruKAN</sub>
    </td>
  </tr>
</table>

## Features

- [x] Compiler
- [x] Plotter
    - [x] TruKAN
    - [x] CustomKAN (ChebyKAN, EfficientKAN, SineKAN)
- [ ] Pruner
    - [x] TruKAN
    - [ ] CustomKAN

## Compiler

- Compile a SymPy symbolic expression into a TruKan model.

### Design differences vs. pykan's compiler
- Pykan pins each edge to a symbolic function using ``fix_symbolic``, which writes to a parallel Symbolic_KANLayer.  TruKan has no symbolic layer. Instead, we *fit* the truncated-power-basis coefficients to the target 1-D function via ordinary least squares over ``knots_range``.  The fit quality depends on how well a degree-``degree`` truncated-power spline can represent the function; cubic (degree=3) is accurate for smooth elementary functions with a reasonably dense knot grid.

- Pykan stores per-node and per-subnode affine parameters (``node_scale``, ``node_bias``, ``subnode_scale``, ``subnode_bias``) as learnable tensors. TruKan only has ``bias_out`` per layer, and the constant basis function (the '1' in [1, u, u², u³, …]) already absorbs constant offsets. We fold *every* affine transformation extracted during tree-parsing directly into the least-squares target function so the compiled model is self-contained and needs no extra parameters.

- Standard TruKan is a sum-only KAN: the output of each layer is```
   output_j = Σ_i  basis(x_i) @ coeffs[i, j, :]   ```
Expressions with products of *distinct symbolic variables* (e.g. x*y, sin(x)*cos(y)) cannot be represented and are rejected.  See the section "Adding multiply nodes to TruKan" below for how to extend the model.

## Generic Plotter & Pruner
- Why it is generic?
  - **trukan** tools work with *any* KAN-style model that follows a minimal interface:
    - Consistent naming convention.
    - Implementation of a single method that returns the numerical activation functions.

- How it works?
  - Any compatible KAN variation can be plotted automatically. No forking or rewriting of visualization code is required.


## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@ARTICLE{Bayeh2026TruKAN,
  title         = "{TruKAN}: Towards more efficient {Kolmogorov-Arnold}
                   networks using truncated power functions",
  author        = "Bayeh, Ali and Sadaoui, Samira and Mouhoub, Malek",  
  month         =  feb,
  year          =  2026,
  copyright     = "http://creativecommons.org/licenses/by/4.0/",
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "2602.03879"
}
```

## License

`trukan` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
