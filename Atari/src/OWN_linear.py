import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable
from sklearn.decomposition import PCA
from typing import Optional
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

def safe_pinverse(W, eps=1e-6):
    if torch.any(torch.isnan(W)) or torch.any(torch.isinf(W)):
        W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        W_inv = torch.pinverse(W, rcond=eps)
    except RuntimeError:
        W_inv = torch.pinverse(W, rcond=eps * 0.1)
    return W_inv

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
        own_iter: int = 6,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.use_own = use_own
        self.eps = eps 
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
        W = self.whitener(self.weight) if do_whiten else self.weight

        if invert:

            y = input - self.bias
            if W.size(0) == W.size(1):

                W_inv = torch.inverse(W)
            else:
      
                try:
                    W_inv = torch.pinverse(W, rcond=self.eps)
                except RuntimeError:
           
                    W_inv = safe_pinverse(W).to(W.device).type_as(W)
         
            return F.linear(y, W_inv.t(), bias=None)

        return F.linear(input, W.t(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.size(0)}, "
            f"out_features={self.weight.size(1)}, "
            f"use_own={self.use_own}"
        )        



if __name__ == '__main__':
    torch.manual_seed(1)
    own = OWNLinear(5, 7)
    x = torch.rand(3,3,1,1)
    print('x',x)
    a = torch.randint(0,1,(3,))
    # print(a)
    batch_range = torch.arange(a.shape[0])
    action_onehot = torch.zeros(a.shape[0],
                                2,
                                x.shape[-2],
                                x.shape[-1])
    action_onehot[batch_range, a, :, :] = 1
    xin = torch.cat([action_onehot, x],dim=1)
    print('xin',xin)
    x1 = own(xin,invert=False)
    x2 = own(x1,invert=True)
    print('invert',x2)
