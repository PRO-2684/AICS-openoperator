import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, offset: int = 0):
        super().__init__(); self.offset = offset
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(A, offset=self.offset, dim1=-2, dim2=-1)
batch = 32; N = 512
def get_inputs(): return [torch.randn(batch, N, N)]
def get_init_inputs(): return [0]
