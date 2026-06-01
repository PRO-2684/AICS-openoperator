import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, output_size):
        super().__init__(); self.output_size = output_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(x, self.output_size)
batch = 4; C = 16; D = 32; H = 32; W = 32
def get_inputs(): return [torch.randn(batch, C, D, H, W)]
def get_init_inputs(): return [(4, 8, 8)]
