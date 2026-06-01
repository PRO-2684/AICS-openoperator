import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, fill_value: float = 0.0):
        super().__init__(); self.fill_value = fill_value
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(mask, self.fill_value)
batch = 32; seq = 512; d = 256
def get_inputs():
    x = torch.randn(batch, seq, d)
    mask = torch.rand(batch, seq, d) > 0.8
    return [x, mask]
def get_init_inputs(): return [float('-inf')]
