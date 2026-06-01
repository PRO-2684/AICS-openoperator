import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
batch = 4; C = 64; H = 64; W = 64
def get_inputs(): return [torch.randn(batch, C, H, W)]
def get_init_inputs(): return [2]
