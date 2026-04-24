import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 2, padding: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=padding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
batch = 16; C = 64; H = 128; W = 128; out_C = 128
def get_inputs(): return [torch.randn(batch, C, H, W)]
def get_init_inputs(): return [64, 128, 3, 2, 2]
