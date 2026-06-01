import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A * B
M = 2048; N = 2048
def get_inputs(): return [torch.randn(M, N), torch.randn(M, N)]
def get_init_inputs(): return []
