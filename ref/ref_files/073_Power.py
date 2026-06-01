import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, exponent: float = 2.0):
        super().__init__(); self.exponent = exponent
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x.abs(), self.exponent)
batch_size = 16; dim = 16384
def get_inputs(): return [torch.randn(batch_size, dim)]
def get_init_inputs(): return [2.0]
