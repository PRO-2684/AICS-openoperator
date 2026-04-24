import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
    def forward(self, x: torch.Tensor):
        return self.pool(x)
batch = 16; channels = 32; H = 128; W = 128
def get_inputs(): return [torch.randn(batch, channels, H, W)]
def get_init_inputs(): return [2, 2]
