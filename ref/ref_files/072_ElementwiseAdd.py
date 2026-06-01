import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b
M = 1024; N = 4096
def get_inputs(): return [torch.randn(M, N), torch.randn(M, N)]
def get_init_inputs(): return []
