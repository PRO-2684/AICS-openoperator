import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                              stride=stride, padding=padding, groups=in_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
batch = 16; C = 64; H = 128; W = 128
def get_inputs(): return [torch.randn(batch, C, H, W)]
def get_init_inputs(): return [64, 3, 1, 1]
