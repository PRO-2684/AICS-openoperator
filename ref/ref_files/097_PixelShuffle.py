import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(x)
batch = 4; C = 64; H = 32; W = 32; r = 2
def get_inputs(): return [torch.randn(batch, C * r * r, H, W)]
def get_init_inputs(): return [2]
