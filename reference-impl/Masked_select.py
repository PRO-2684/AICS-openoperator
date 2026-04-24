import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, input: torch.Tensor, threshold: float) -> torch.Tensor:
        mask = input > threshold
        return torch.masked_select(input, mask)
M = 1024; N = 1024
def get_inputs(): return [torch.randn(M, N), 0.0]
def get_init_inputs(): return []
