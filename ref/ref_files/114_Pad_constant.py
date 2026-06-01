import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, pad, value: float = 0.0):
        super().__init__(); self.pad = pad; self.value = value
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.pad, mode='constant', value=self.value)
batch = 16; C = 64; H = 128; W = 128
def get_inputs(): return [torch.randn(batch, C, H, W)]
def get_init_inputs(): return [(2, 2, 2, 2), 0.0]
