import torch, torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 2, padding: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv.weight.data = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


batch = 16
C = 64
H = 128
W = 128
out_C = 128
kernel_size = 3
dilation = 2
padding = 2


def get_inputs():
    return [torch.randn(batch, C, H, W)]


def get_init_inputs():
    kernel = torch.randn(out_C, C, kernel_size, kernel_size)
    return [kernel, C, out_C, kernel_size, dilation, padding]
