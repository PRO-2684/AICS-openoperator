import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def forward(self, input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
batch = 4; C = 64; H = 128; W = 128; out_H = 64; out_W = 64
def get_inputs():
    inp = torch.randn(batch, C, H, W)
    grid = torch.rand(batch, out_H, out_W, 2) * 2 - 1
    return [inp, grid]
def get_init_inputs(): return []
