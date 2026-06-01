import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(output_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
batch = 16; channels = 64; H = 256; W = 256; out_h = 8; out_w = 8
def get_inputs(): return [torch.randn(batch, channels, H, W)]
def get_init_inputs(): return [(out_h, out_w)]
