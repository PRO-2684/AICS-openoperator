import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(1, 2))
batch_size = 16; H = 256; W = 256
def get_inputs(): return [torch.randn(batch_size, H, W)]
def get_init_inputs(): return []
