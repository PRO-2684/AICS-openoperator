import torch
import torch.nn as nn


M = 256
K = 512
N = 1024
DTYPE = torch.float16
SEED = 2026
RAND_LOW = -5
RAND_HIGH = 6


class Model(nn.Module):
    """
    Reference implementation of the fused matmul forward path with optional bias.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes ``x @ w`` and optionally adds ``b`` row-wise.

        Args:
            x: Input matrix of shape (M, K).
            w: Weight matrix of shape (K, N).
            b: Optional bias vector of shape (N,).

        Returns:
            Output tensor of shape (M, N).
        """
        out = torch.matmul(x.float(), w.float())
        if b is not None:
            out = out + b.float()
        return out.to(x.dtype)


def get_inputs():
    torch.manual_seed(SEED)
    x = torch.randint(RAND_LOW, RAND_HIGH, (M, K), dtype=torch.int32).to(DTYPE)
    w = torch.randint(RAND_LOW, RAND_HIGH, (K, N), dtype=torch.int32).to(DTYPE)
    b = torch.randint(RAND_LOW, RAND_HIGH, (N,), dtype=torch.int32).to(DTYPE)
    return [x, w, b]


def get_init_inputs():
    return []
