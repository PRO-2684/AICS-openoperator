import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, k: int, dim: int):
        super().__init__(); self.k = k; self.dim = dim
    def forward(self, x: torch.Tensor):
        return torch.topk(x, self.k, dim=self.dim)
batch_size = 16; dim = 1024
def get_inputs(): return [torch.randn(batch_size, dim)]
def get_init_inputs(): return [10, 1]
