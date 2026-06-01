import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unfold(x)
batch = 16; C = 3; H = 256; W = 256; K = 3
def get_inputs(): return [torch.randn(batch, C, H, W)]
def get_init_inputs(): return [3, 1, 0]
