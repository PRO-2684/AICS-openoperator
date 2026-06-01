import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor):
        return torch.linalg.qr(A)
batch = 8; M = 128; N = 64
def get_inputs(): return [torch.randn(batch, M, N)]
def get_init_inputs(): return []
