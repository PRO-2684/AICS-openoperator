import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x)
batch_size = 64; dim = 8192
def get_inputs(): return [torch.randn(batch_size, dim)]
def get_init_inputs(): return []
