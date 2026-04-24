import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cummin(x, dim=self.dim).values
batch_size = 128; length = 4000
def get_inputs(): return [torch.randn(batch_size, length)]
def get_init_inputs(): return [1]
