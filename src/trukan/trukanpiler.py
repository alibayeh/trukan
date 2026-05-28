from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import sympy
import torch
from sympy import (
    Abs,
    Add,
    Mul,
    Pow,
    Symbol,
    acos,
    asin,
    atan,
    atanh,
    cos,
    exp,
    lambdify,
    log,
    sign,
    sin,
    sqrt,
    symbols,
    tan,
    tanh,
)

from .trukan import TruKan
from .trukan_layer import truncated_power_basis, truncated_power_basis_d3


def _next_nontrivial(expr, scale=1, bias=0):
    if expr.func in (Add, Mul):
        nums, vars_ = [], []
        for a in expr.args:
            (nums if a.is_number else vars_).append(a)
        if nums:
            if expr.func == Add:
                bias = sum(nums)
            else:  # Mul
                scale = sympy.Mul(*nums)
            return _next_nontrivial(expr.func(*vars_), scale, bias)
    return expr, scale, bias


_SYMPY_TO_NP: dict[object, Callable] = {
    exp: np.exp,
    sin: np.sin,
    cos: np.cos,
    tan: np.tan,
    sqrt: np.sqrt,
    log: np.log,
    tanh: np.tanh,
    asin: np.arcsin,
    acos: np.arccos,
    atan: np.arctan,
    atanh: np.arctanh,
    Abs: np.abs,
    sign: np.sign,
}


def _to_numpy(func, power_exponent=None) -> Callable:
    if func in _SYMPY_TO_NP:
        return _SYMPY_TO_NP[func]
    if func is Pow:
        alpha = float(power_exponent)
        # Use abs+sign to stay real-valued for negative inputs when alpha is integer
        if alpha == int(alpha):
            return lambda x, _a=int(alpha): x**_a
        return lambda x, _a=alpha: np.abs(x) ** _a * np.sign(x)
    _x = symbols("_x_compiler_")
    return lambdify(_x, func(_x), "numpy")


def _identity(x):
    return x


def fit_basis_to_function(
    numpy_fun: Callable,
    affine: list[float],
    knots: torch.Tensor,
    degree: int,
    knots_range: tuple[float, float],
    n_samples: int,
) -> torch.Tensor:
    a, b, c, d = [float(v) for v in affine]
    lo, hi = knots_range
    x_np = np.linspace(lo, hi, n_samples, dtype=np.float64)
    try:
        y_np = c * numpy_fun(a * x_np + b) + d
    except Exception:
        y_np = np.array(
            [c * float(numpy_fun(float(a * xi + b))) + d for xi in x_np],
            dtype=np.float64,
        )

    y_np = np.asarray(y_np, dtype=np.float64).ravel()
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(-1)
    knots_cpu = knots.cpu()
    k1 = knots_cpu.unsqueeze(0)
    if degree == 3:
        B_t = truncated_power_basis_d3(x_t, k1)
    else:
        exps = torch.arange(degree + 1, dtype=torch.float32)
        B_t = truncated_power_basis(x_t, exps, degree, k1)

    B = B_t[:, 0, :].double().numpy()
    w, *_ = np.linalg.lstsq(B, y_np, rcond=None)
    return torch.tensor(w, dtype=torch.float32)


class _Node:
    """
    A node in the expression tree, representing the result of a sub-expression.

    Depth 0 = root (final output).
    Depth n_layer = leaves (input variables).
    """

    def __init__(
        self,
        expr,
        depth: int,
        scale,
        bias,
        Nodes: list[list],
        parent: _SubNode | None = None,
    ):
        self.expr = expr
        self.depth = depth
        self.scale = float(scale)
        self.bias = float(bias)

        # Register in the global Nodes list
        while len(Nodes) <= depth:
            Nodes.append([])
        self.index: int = len(Nodes[depth])
        Nodes[depth].append(self)
        # Back-link to parent SubNode
        self.parent_index: int | None = parent.index if parent is not None else None
        # Forward links (filled later) – list of SubNode indices
        self.child_indices: list[int] = []


class _SubNode:
    """
    A summation point that aggregates incoming activations
    before passing them to the parent _Node.
    One _SubNode is created for each _Node that is not a leaf.
    """

    def __init__(
        self,
        expr,
        depth: int,
        scale,
        bias,
        SubNodes: list[list],
        parent_node: _Node,
    ):
        self.expr = expr
        self.depth = depth
        self.scale = float(scale)
        self.bias = float(bias)

        while len(SubNodes) <= depth:
            SubNodes.append([])
        self.index: int = len(SubNodes[depth])
        SubNodes[depth].append(self)
        self.parent_index: int = parent_node.index
        self.child_indices: list[int] = []
        parent_node.child_indices.append(self.index)


class _Connection:
    """
    An activation-function edge connecting a child _Node to a parent _SubNode.
    """

    def __init__(
        self,
        affine: list[float],
        numpy_fun: Callable,
        fun_name: str,
        subnode: _SubNode,
        child_node: _Node,
        Connections: dict,
    ):
        self.affine = list(affine)
        self.numpy_fun = numpy_fun
        self.fun_name = fun_name
        self.depth = subnode.depth
        self.subnode_idx = subnode.index
        self.node_idx = child_node.index
        Connections[(self.depth, self.subnode_idx, self.node_idx)] = self
        subnode.child_indices.append(child_node.index)


def _build_tree(
    expr_root,
    Nodes: list[list[_Node]],
    SubNodes: list[list[_SubNode]],
    Connections: dict,
    Start_Nodes: list[_Node],
    n_layer_target: int | None,
) -> None:
    """
    Recursively parse expr_root into the shared tree structures.

    Called twice:
      - Pass 1 (n_layer_target=None): discovers tree depth.
      - Pass 2 (n_layer_target=n_layer): pads leaf nodes with identity layers.
    """

    def mk_node(expr, depth, scale, bias, parent_sn):
        return _Node(expr, depth, scale, bias, Nodes, parent=parent_sn)

    def mk_subnode(expr, depth, scale, bias, parent_node):
        sn = _SubNode(expr, depth, scale, bias, SubNodes, parent_node=parent_node)
        return sn

    def connect(affine, fn, fn_name, subnode, child_node):
        _Connection(affine, fn, fn_name, subnode, child_node, Connections)

    def _check_no_var_product(term):
        """
        Raise a clear error if term is a Mul of distinct symbolic variables.
        """
        if term.func == Mul:
            non_num = [a for a in term.args if not a.is_number]
            if len(non_num) > 1:
                raise ValueError(
                    f"\n\nProduct of distinct symbolic variables detected: {term}\n"
                    "Standard TruKan cannot represent this.\n\n"
                    "Option:\n"
                    "  Rewrite the expression using only sums and 1-D functions.\n"
                    "  e.g. x*y can be approximated as ((x+y)²-x²-y²)/2 but\n"
                    "  this requires a 2-layer network and may not be exact.\n"
                )

    def create_node(expr, parent_sn: _SubNode | None, depth: int) -> _Node:
        """
        Create a tree node for expr at the given depth.
        parent_sn is the _SubNode this node will feed into (None for root).
        """
        expr, scale, bias = _next_nontrivial(expr)

        # LEAF: input Symbol
        if expr.func == Symbol:
            node = mk_node(expr, depth, scale, bias, parent_sn)
            node.is_leaf = True
            if n_layer_target is not None:
                # Pad with identity connections so all leaves reach n_layer_target
                cur = node
                for _ in range(n_layer_target - depth):
                    pad_sn = mk_subnode(expr, cur.depth + 1, 1.0, 0.0, cur)
                    nxt = mk_node(expr, cur.depth + 1, 1.0, 0.0, pad_sn)
                    connect([1.0, 0.0, 1.0, 0.0], _identity, "x", pad_sn, nxt)
                    cur = nxt
            Start_Nodes.append(node)
            return node

        # 1-D function: sin, exp, Pow, log,...
        elif expr.func not in (Add, Mul):
            node = mk_node(expr, depth, scale, bias, parent_sn)
            power_exp = expr.args[1] if expr.func is Pow else None
            inner = expr.args[0]
            inner_e, sc2, bi2 = _next_nontrivial(inner)
            sn = mk_subnode(inner_e, depth + 1, sc2, bi2, node)
            child = create_node(inner, sn, depth + 1)
            connect(
                [1.0, 0.0, 1.0, 0.0],
                _to_numpy(expr.func, power_exp),
                str(expr.func),
                sn,
                child,
            )
            return node

        # Add node: sum of terms
        elif expr.func == Add:
            node = mk_node(expr, depth, scale, bias, parent_sn)
            sn = mk_subnode(expr, depth + 1, 1.0, 0.0, node)

            for term in expr.args:
                term, sc2, bi2 = _next_nontrivial(term)
                _check_no_var_product(term)

                if term.func == Symbol:
                    # Direct variable (linear identity activation)
                    child = create_node(term, sn, depth + 1)
                    connect(
                        [1.0, 0.0, float(sc2), float(bi2)], _identity, "x", sn, child
                    )

                elif term.func == Mul:
                    # any remaining Mul has at most one non-numeric factor.
                    non_num = [a for a in term.args if not a.is_number]
                    child = create_node(non_num[0], sn, depth + 1)
                    connect(
                        [1.0, 0.0, float(sc2), float(bi2)], _identity, "x", sn, child
                    )

                elif term.func not in (Add, Mul):
                    # 1-D function inside Add: f(inner)
                    power_exp = term.args[1] if term.func is Pow else None
                    inner = term.args[0]
                    child = create_node(inner, sn, depth + 1)
                    connect(
                        [1.0, 0.0, float(sc2), float(bi2)],
                        _to_numpy(term.func, power_exp),
                        str(term.func),
                        sn,
                        child,
                    )
                else:
                    raise ValueError(f"Unsupported term structure inside Add: {term}")

            return node

        # Mul with a single variable factor (constant × variable) ───────
        elif expr.func == Mul:
            non_num = [a for a in expr.args if not a.is_number]
            _check_no_var_product(expr)  # raises if >1 symbolic factor
            return create_node(non_num[0], parent_sn, depth)

        else:
            raise ValueError(f"Unsupported SymPy node type: {type(expr)}: {expr}")

    create_node(expr_root, parent_sn=None, depth=0)


def expr2truKAN(
    input_variables: Sequence,
    expr,
    num_knots: int = 5,
    degree: int = 3,
    knots_range: tuple[float, float] = (-1.0, 1.0),
    n_fit_samples: int = 500,
    learn_knots: bool = False,
    shared_knots: bool = True,
    seed: int = 1,
    device=None,
) -> TruKan:
    device = device or "cpu"
    Nodes1: list[list] = [[]]
    SN1: list[list] = [[]]
    C1: dict = {}
    SL1: list = []
    _build_tree(expr, Nodes1, SN1, C1, SL1, n_layer_target=None)
    n_layer = len(Nodes1) - 1

    if n_layer == 0:
        n_layer = 1

    Nodes: list[list[_Node]] = [[]]
    SubNodes: list[list[_SubNode]] = [[]]
    Connections: dict = {}
    Start_Nodes: list[_Node] = []
    _build_tree(expr, Nodes, SubNodes, Connections, Start_Nodes, n_layer_target=n_layer)
    n_layer = len(Nodes) - 1

    for leaf in Start_Nodes:
        key = (leaf.depth, leaf.parent_index, leaf.index)
        if key in Connections:
            conn = Connections[key]
            conn.affine[0] = leaf.scale
            conn.affine[1] = leaf.bias
            leaf.scale = 1.0
            leaf.bias = 0.0

    sym2var: dict[object, int] = {v: i for i, v in enumerate(input_variables)}
    node2var: dict[tuple[int, int], int] = {}
    for leaf in Start_Nodes:
        vi = sym2var.get(leaf.expr)
        if vi is None:
            raise ValueError(
                f"Symbol '{leaf.expr}' appears in the expression but was not "
                f"found in input_variables={list(input_variables)}."
            )
        node2var[(leaf.depth, leaf.index)] = vi

    node_kan: dict[tuple[int, int], int] = {}
    sn_kan: dict[tuple[int, int], int] = {}

    for d, level in enumerate(Nodes):
        for node in level:
            if d == n_layer:
                node_kan[(d, node.index)] = node2var[(d, node.index)]
            else:
                node_kan[(d, node.index)] = node.index

    for d, level in enumerate(SubNodes):
        for sn in level:
            sn_kan[(d, sn.index)] = sn.index

    layers_hidden = [len(input_variables)]
    for _l in range(1, n_layer + 1):
        tree_d = n_layer - _l
        layers_hidden.append(len(Nodes[tree_d]) if tree_d < len(Nodes) else 1)

    model = TruKan(
        layers_hidden=layers_hidden,
        num_knots=num_knots,
        degree=degree,
        learn_knots=learn_knots,
        shared_knots=shared_knots,
        knots_range=knots_range,
        seed=seed,
        device=device,
    )

    with torch.no_grad():
        for layer in model.layers.values():
            layer.coeffs.data.zero_()
            layer.bias_out.data.zero_()

    layer_keys = list(model.layers.keys())

    with torch.no_grad():
        for (depth, sn_idx, n_idx), conn in Connections.items():
            _l = n_layer - depth
            if _l < 0 or _l >= len(layer_keys):
                continue  # safety guard

            _in = node_kan.get((depth, n_idx), n_idx)
            _out = sn_kan.get((depth, sn_idx), sn_idx)

            layer = model.layers[layer_keys[_l]]

            if _in >= layer.in_dim or _out >= layer.out_dim:
                raise IndexError(
                    f"Index out of bounds for TruKanLayer_{_l}: "
                    f"trukan_in={_in} (in_dim={layer.in_dim}), "
                    f"trukan_out={_out} (out_dim={layer.out_dim}). "
                    "This likely indicates a mismatch between the expression "
                    "tree and layers_hidden. Please report this as a bug."
                )

            knots_all = layer.get_knots()
            if shared_knots:
                knots_ij = knots_all[_in].cpu()
            else:
                knots_ij = knots_all[_in, _out].cpu()

            w = fit_basis_to_function(
                numpy_fun=conn.numpy_fun,
                affine=conn.affine,
                knots=knots_ij,
                degree=degree,
                knots_range=knots_range,
                n_samples=n_fit_samples,
            )

            layer.coeffs.data[_in, _out, :] = w.to(device)

    return model


trukanpiler = expr2truKAN


def validate_compiled_model(
    input_variables: Sequence,
    expr,
    model: TruKan,
    n_test: int = 200,
    knots_range: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, float]:
    import numpy as np
    from sympy import lambdify as sy_lambdify

    n_in = len(input_variables)
    lo, hi = knots_range

    # Random test inputs
    rng = np.random.default_rng(42)
    x_np = rng.uniform(lo, hi, size=(n_test, n_in))

    # SymPy ground truth
    args = [x_np[:, i] for i in range(n_in)]
    try:
        f_np = sy_lambdify(input_variables, expr, "numpy")
        y_true = np.asarray(f_np(*args), dtype=np.float64).ravel()
    except Exception as e:
        return {"error": str(e)}

    # TruKan prediction
    x_t = torch.tensor(x_np, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_t).squeeze(-1).cpu().numpy().astype(np.float64)

    abs_err = np.abs(y_true - y_pred)
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    denom = float(np.sqrt(np.mean(y_true**2))) + 1e-12
    rel_rmse = rmse / denom

    results = {
        "max_abs_error": float(abs_err.max()),
        "rmse": rmse,
        "rel_rmse": rel_rmse,
    }
    return results
