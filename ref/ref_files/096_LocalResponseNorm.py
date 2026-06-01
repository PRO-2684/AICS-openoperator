import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
        super().__init__()
        self.lrn = nn.LocalResponseNorm(size, alpha=alpha, beta=beta, k=k)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrn(x)
batch = 16; channels = 64; H = 56; W = 56
def get_inputs(): return [torch.randn(batch, channels, H, W)]
def get_init_inputs(): return [5]
