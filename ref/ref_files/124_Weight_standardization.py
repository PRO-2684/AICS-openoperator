import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        mean = w.mean(dim=[1, 2, 3], keepdim=True)
        std  = w.std(dim=[1, 2, 3], keepdim=True) + 1e-5
        w_std = (w - mean) / std
        return F.conv2d(x, w_std, padding=1)
batch = 8; C = 64; H = 128; W = 128
def get_inputs(): return [torch.randn(batch, C, H, W)]
def get_init_inputs(): return [64, 64, 3]
