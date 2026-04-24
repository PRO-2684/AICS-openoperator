import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bi,bj->bij', a, b)
batch = 64; M = 256; N = 256
def get_inputs(): return [torch.randn(batch, M), torch.randn(batch, N)]
def get_init_inputs(): return []
