import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        h = x + residual
        rms = h.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return h * rms * self.weight
batch = 32; seq = 128; d_model = 512
def get_inputs(): return [torch.randn(batch, seq, d_model), torch.randn(batch, seq, d_model)]
def get_init_inputs(): return [d_model]
