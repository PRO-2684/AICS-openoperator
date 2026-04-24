import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cumsum(x * mask.float(), dim=dim)
batch = 128; length = 4000
def get_inputs():
    x = torch.randn(batch, length)
    mask = torch.rand(batch, length) > 0.3
    return [x, mask, 1]
def get_init_inputs(): return []
