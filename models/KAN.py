import torch
import torch.nn as nn
from .bspline import make_open_uniform_knots, bspline_basis

class KANLayer(nn.Module):
    """
    A KAN layer that maps (B, in_dim) -> (B, out_dim)
    Each edge (out_j <- in_i) is a learnable univariate spline:
        f_{j,i}(x_i) = sum_r c_{j,i,r} * B_r(x_i)
    Then out_j = sum_i f_{j,i}(x_i) + bias_j
    """
    def __init__(self, in_dim: int, out_dim: int, degree: int = 3, num_internal_knots: int = 5,
                 x_min: float = -1.0, x_max: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.num_internal_knots = num_internal_knots

        # knot vector shared across all edges
        knots = make_open_uniform_knots(x_min, x_max, num_internal_knots, degree)
        self.register_buffer("knots", knots)

        n_basis = knots.numel() - degree - 1
        # coefficients: (out_dim, in_dim, n_basis)
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, n_basis) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor):
        # x: (B, in_dim)
        B = x.size(0)
        out = x.new_zeros((B, self.out_dim))

        # compute basis per input dimension
        for i in range(self.in_dim):
            basis = bspline_basis(x[:, i], self.knots, self.degree)  # (B, n_basis)
            # (out_dim, n_basis) @ (B, n_basis)^T -> (out_dim, B) -> (B, out_dim)
            # using einsum: out[:, j] += sum_r coeff[j,i,r] * basis[:,r]
            out += torch.einsum("br,oir->bo", basis, self.coeff[:, i, :].unsqueeze(1).transpose(0,1)).squeeze(1)

        out = out + self.bias
        return out

class KANNet(nn.Module):
    def __init__(self, dims, degree=3, num_internal_knots=5, x_min=-1.0, x_max=1.0):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(KANLayer(a, b, degree=degree, num_internal_knots=num_internal_knots,
                                   x_min=x_min, x_max=x_max))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return x
