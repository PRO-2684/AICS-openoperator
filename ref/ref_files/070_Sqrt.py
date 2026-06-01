import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(x))
batch_size = 16; dim = 16384
def get_inputs(): return [torch.randn(batch_size, dim)]
def get_init_inputs(): return []
