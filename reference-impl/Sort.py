import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, dim: int, descending: bool = False):
        super().__init__(); self.dim = dim; self.descending = descending
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sort(x, dim=self.dim, descending=self.descending).values
batch_size = 16; dim = 1024
def get_inputs(): return [torch.randn(batch_size, dim)]
def get_init_inputs(): return [1, False]
