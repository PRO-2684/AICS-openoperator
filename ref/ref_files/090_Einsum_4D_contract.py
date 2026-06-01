import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bhid,bhjd->bhij', A, B)
batch = 4; heads = 8; seq = 128; dim = 64
def get_inputs():
    A = torch.randn(batch, heads, seq, dim)
    B = torch.randn(batch, heads, seq, dim)
    return [A, B]
def get_init_inputs(): return []
