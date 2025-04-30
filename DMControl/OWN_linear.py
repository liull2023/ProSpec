import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable
from typing import Optional
# from sklearn.decomposition import PCA

#  norm funcitons--------------------------------
class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input
    
def _matrix_inv_sqrt_newton(A: torch.Tensor, num_iter: int = 6) -> torch.Tensor:
    """
    Compute A^{-1/2} using Newton-Schulz iteration.
    A must be symmetric positive definite.
    """
    dim = A.shape[-1]
    I = torch.eye(dim, device=A.device, dtype=A.dtype).expand_as(A)
    normA = A.norm(dim=(-2, -1), keepdim=True)
    Y = A / normA
    Z = I.clone()
    for _ in range(num_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z / torch.sqrt(normA)


class OWNNorm(nn.Module):
    """
    Orthogonal weight normalization via approximate inverse sqrt of covariance.
    Uses Newton-Schulz iteration to avoid expensive SVD/eigendecomp.
    Splits weights into groups, centers them, computes covariance,
    and applies approximate whitening: W_norm = S^{-1/2} (W - mean).
    """

    def __init__(self, num_groups: int = 1, num_iter: int = 6) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_iter = num_iter

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        C, _ = weight.shape
        assert C % self.num_groups == 0, \
            f"Channels ({C}) must be divisible by num_groups ({self.num_groups})."
        group_size = C // self.num_groups
        Z = weight.view(self.num_groups, group_size, -1)
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        cov = Zc @ Zc.transpose(1, 2)
        inv_sqrt = _matrix_inv_sqrt_newton(cov, num_iter=self.num_iter)
        W_norm = (inv_sqrt @ Zc).reshape_as(weight)
        return W_norm

    def extra_repr(self) -> str:
        return f"OWNNorm(num_groups={self.num_groups}, num_iter={self.num_iter})"


class OWNLinear(nn.Module):
    """
    Linear layer with optional orthogonal weight normalization.

    Supports exact inversion when weight is square.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_own: bool = True,
        num_groups: int = 1,
        own_iter: int = 6
    ) -> None:
        super().__init__()
        self.use_own = use_own
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.whitener = OWNNorm(num_groups, num_iter=own_iter)

    def forward(
        self,
        input: torch.Tensor,
        apply_own: Optional[bool] = None,
        invert: bool = False
    ) -> torch.Tensor:
        do_whiten = self.use_own if apply_own is None else apply_own
        W_used = self.whitener(self.weight) if do_whiten else self.weight
        if invert:
            y_centered = input - self.bias
            if W_used.size(0) == W_used.size(1):
                W_inv = torch.inverse(W_used)
            else:
                W_inv = torch.pinverse(W_used)
            return y_centered @ W_inv
        return input @ W_used + self.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.size(0)}, "
            f"out_features={self.weight.size(1)}, "
            f"use_own={self.use_own}"
        )


if __name__ == '__main__':
    torch.manual_seed(0)
    layer = OWNLinear(4, 4, num_groups=1, own_iter=6)
    x = torch.randn(8, 4)
    print(f'orign:{x}')
    # Warm-up
    for _ in range(3): _ = layer(x)

    # Forward
    y = layer(x)
    print(f'Forward:{y}')
    print('Output:', y.shape)
    # Inverse
    x_rec = layer(y, invert=True)
    print('Reconstructed x:', x_rec.shape)
    print(f'Inverse:{x_rec}')