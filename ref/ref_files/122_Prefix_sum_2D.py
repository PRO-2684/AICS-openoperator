import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(torch.cumsum(x, dim=1), dim=2)
batch = 16; H = 256; W = 256
def get_inputs(): return [torch.randn(batch, H, W)]
def get_init_inputs(): return []
