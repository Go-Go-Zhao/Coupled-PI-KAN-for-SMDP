import torch
import torch.nn as nn

def make_open_uniform_knots(x_min: float, x_max: float, num_internal: int, degree: int, device=None):
    """
    Open-uniform knot vector:
    [x_min repeated degree+1] + internal knots + [x_max repeated degree+1]
    """
    internal = torch.linspace(x_min, x_max, steps=num_internal + 2, device=device)[1:-1]
    left = torch.full((degree + 1,), x_min, device=device)
    right = torch.full((degree + 1,), x_max, device=device)
    knots = torch.cat([left, internal, right], dim=0)
    return knots  # shape: (num_internal + 2*(degree+1),)

def bspline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int):
    """
    Cox–de Boor recursion to compute all B-spline basis values at x.
    x: (B,) or (B,1)
    knots: (K,)
    returns: (B, n_basis)
    where n_basis = K - degree - 1
    """
    if x.dim() == 2 and x.size(-1) == 1:
        x = x.squeeze(-1)
    B = x.shape[0]
    K = knots.numel()
    n_basis = K - degree - 1

    # degree 0
    # N_{i,0}(x) = 1 if t_i <= x < t_{i+1} else 0
    t_i = knots[:-1].unsqueeze(0).expand(B, K - 1)
    t_ip1 = knots[1:].unsqueeze(0).expand(B, K - 1)
    x_ = x.unsqueeze(1).expand(B, K - 1)
    N = ((t_i <= x_) & (x_ < t_ip1)).float()

    # include the right boundary x==x_max
    x_max = knots[-1]
    N = torch.where((x == x_max).unsqueeze(1), (t_i <= x_).float() * (x_ <= t_ip1).float(), N)

    # elevate degree
    for k in range(1, degree + 1):
        N_new = torch.zeros((B, K - 1 - k), device=x.device, dtype=x.dtype)
        for i in range(K - 1 - k):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]
            term1 = 0.0
            term2 = 0.0
            if denom1.abs() > 0:
                term1 = (x - knots[i]) / denom1 * N[:, i]
            if denom2.abs() > 0:
                term2 = (knots[i + k + 1] - x) / denom2 * N[:, i + 1]
            N_new[:, i] = term1 + term2
        N = N_new

    # N shape: (B, n_basis)
    return N[:, :n_basis]
