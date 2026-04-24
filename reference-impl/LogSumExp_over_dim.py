import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x, dim=self.dim)
batch_size = 16; dim = 1024
def get_inputs(): return [torch.randn(batch_size, dim)]
def get_init_inputs(): return [1]
