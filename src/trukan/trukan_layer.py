import torch
import torch.nn as nn
import torch.nn.functional as F


def truncated_power_basis_d3(u: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    u = u.contiguous()
    ones = torch.ones_like(u)
    u2 = u.square()
    u3 = u2 * u
    poly = torch.stack([ones, u, u2, u3], dim=-1)
    diff = u.unsqueeze(-1) - knots
    trunc = torch.relu(diff).pow(3)
    return torch.cat([poly, trunc], dim=-1)


def truncated_power_basis(
    u: torch.Tensor,
    exponents: torch.Tensor,
    degree: int,
    knots: torch.Tensor,
) -> torch.Tensor:
    u_exp = u.unsqueeze(-1)  # (batch, in_dim, 1)
    poly = u_exp.pow(exponents)  # (batch, in_dim, degree+1)
    knots_b = knots.unsqueeze(0)  # (1, in_dim, num_knots)
    diff = u_exp - knots_b  # (batch, in_dim, num_knots)
    trunc = torch.relu(diff).pow(degree)  # (batch, in_dim, num_knots)
    return torch.cat([poly, trunc], dim=-1)  # (batch, in_dim, basis_size)


class TruKanLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        degree: int = 3,
        num_knots: int = 6,
        learn_knots: bool = False,
        shared_knots: bool = True,
        knots_range: tuple[float, float] = (-1.0, 1.0),
        eps: float = 1e-8,
        device=None,
        name: str = "",
    ):
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.degree = int(degree)
        self.num_knots = int(num_knots)
        self.learn_knots = bool(learn_knots)
        self.shared_knots = bool(shared_knots)  # Only for forward_traced
        self.eps = eps
        self.name = name if len(name) > 0 else None  # Only for forward_traced

        self.basis_size = (self.degree + 1) + self.num_knots
        # self.basis_size = self.num_knots

        self.coeffs = nn.Parameter(torch.randn(in_dim, out_dim, self.basis_size) * 1e-2)
        self.bias_out = nn.Parameter(torch.zeros(out_dim))

        if self.learn_knots:
            lo, hi = knots_range
            # uniform_knots = torch.linspace(lo, hi, steps=num_knots + 2)[1:-1]
            # init_knots = uniform_knots.unsqueeze(0).expand(in_dim, num_knots)
            positive_increments = (hi - lo) / (num_knots + 1)
            raw_init = torch.log(torch.exp(torch.tensor(positive_increments)) - 1)
            if shared_knots:
                self.raw_knots = nn.Parameter(
                    torch.ones(in_dim, num_knots, device=device) * raw_init
                )

                def get_knots() -> torch.Tensor:
                    positive = F.softplus(self.raw_knots) + self.eps
                    cum_frac = torch.cumsum(positive, dim=1)
                    cum_frac = cum_frac / cum_frac[:, -1:].clamp_min(self.eps)
                    knots = lo + (hi - lo) * cum_frac
                    return knots
            else:
                self.raw_knots = nn.Parameter(
                    torch.ones(in_dim, out_dim, num_knots, device=device) * raw_init
                )

                def get_knots() -> torch.Tensor:
                    positive = F.softplus(self.raw_knots) + self.eps
                    cum_frac = torch.cumsum(positive, dim=2)
                    cum_frac = cum_frac / cum_frac[:, :, -1:].clamp_min(self.eps)
                    knots = lo + (hi - lo) * cum_frac
                    return knots
        else:
            uniform_knots = torch.linspace(
                knots_range[0], knots_range[1], steps=num_knots + 2
            )[1:-1]

            if shared_knots:
                fixed_knots = uniform_knots.unsqueeze(0).expand(in_dim, num_knots)
            else:
                # Shape: (in_dim, out_dim, num_knots)
                fixed_knots = uniform_knots.view(1, 1, num_knots).expand(
                    in_dim, out_dim, num_knots
                )

            self.register_buffer("fixed_knots", fixed_knots)

            def get_knots() -> torch.Tensor:
                return self.fixed_knots

        self.get_knots = get_knots  # Only for forward_traced
        if self.degree == 3:

            def compute_basis(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
                return truncated_power_basis_d3(x, knots)

        else:
            self.register_buffer("exponents", torch.arange(degree + 1, device=device))

            def compute_basis(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
                return truncated_power_basis(x, self.exponents, self.degree, knots)

        self.compute_basis = compute_basis  # Only for forward_traced
        if shared_knots:

            def compute_forward(x: torch.Tensor):
                basis = compute_basis(x, get_knots())  # (batch, in_dim, basis_size)
                output = torch.einsum("bid,iod->bo", basis, self.coeffs) + self.bias_out
                return output
        else:

            def compute_forward(x: torch.Tensor):
                # (batch, in_dim, out_dim)
                x_expanded = x.unsqueeze(2).expand(-1, -1, out_dim)
                knots = get_knots()
                # (batch, in_dim, out_dim, basis_size)
                basis = compute_basis(x_expanded, knots)
                output = (
                    torch.einsum("biod,iod->bo", basis, self.coeffs) + self.bias_out
                )
                return output

        self.compute_forward = compute_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.shape[1] == self.in_dim
        # knots = self.get_knots()  # (in_dim, num_knots)
        # basis = self.compute_basis(x)  # (batch, in_dim, basis_size)
        # output = torch.einsum("bid,iod->bo", basis, self.coeffs) + self.bias_out
        return self.compute_forward(x)

    def forward_traced(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.shared_knots:
            basis = self.compute_basis(x, self.get_knots())
            basis_d, basis_n = torch.split(
                basis, [self.degree + 1, self.num_knots], dim=-1
            )
            coeffs_d, coeffs_n = torch.split(
                self.coeffs, [self.degree + 1, self.num_knots], dim=-1
            )

            intermediate_d = basis_d.unsqueeze(2) * coeffs_d.unsqueeze(0)
            intermediate_d = intermediate_d.sum(dim=3)

            intermediate_n = basis_n.unsqueeze(2) * coeffs_n.unsqueeze(0)
            intermediate_n = intermediate_n.sum(dim=3)

            intermediate = intermediate_d + intermediate_n
            output = intermediate.sum(dim=1)

            # output = torch.einsum("bid,iod->bo", basis, self.coeffs) + self.bias_out
            # intermediate = basis.unsqueeze(2) * self.coeffs.unsqueeze(0)
            # intermediate2 = intermediate.sum(dim=3)
            # output2 = intermediate2.sum(dim=1)

            # diff = torch.allclose(output1, output2)
            # print(f"output_1 ≈ output_2? {diff}")
            # max_diff = torch.max(torch.abs(output1 - output2))
            # print(f"Max difference: {max_diff}")
            # diff = torch.allclose(intermediate1, intermediate2)
            # print(f"intermediate1 ≈ intermediate2? {diff}")
            # max_diff = torch.max(torch.abs(intermediate1 - intermediate2))
            # print(f"Max difference: {max_diff}")

            return output, intermediate, intermediate_d, intermediate_n
        else:
            raise Exception(
                "TruKAN with individual knots per output is not supported yet."
            )
            x_expanded = x.unsqueeze(2).expand(-1, -1, self.out_dim)
            knots = self.get_knots()
            basis = self.compute_basis(x_expanded, knots)
            # output = torch.einsum("biod,iod->bo", basis, self.coeffs) + self.bias_out
            intermediate = basis.unsqueeze(2) * self.coeffs.unsqueeze(
                0
            )  # shape: (batch, input, output, basis_size)
            intermediate = intermediate.sum(
                dim=3
            )  # sum over basis_size, shape: (batch, input, output)
            output = intermediate.sum(dim=1)  # sum over input, shape: (batch, output)
            return output, intermediate
