import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.where(condition, x, y)
M = 1024; N = 1024
def get_inputs():
    cond = torch.rand(M, N) > 0.5
    x = torch.randn(M, N)
    y = torch.randn(M, N)
    return [cond, x, y]
def get_init_inputs(): return []
