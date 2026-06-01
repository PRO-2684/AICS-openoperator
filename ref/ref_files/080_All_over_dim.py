import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.all(x > 0, dim=self.dim)
batch_size = 16; dim1 = 256; dim2 = 256
def get_inputs(): return [torch.randn(batch_size, dim1, dim2)]
def get_init_inputs(): return [1]
