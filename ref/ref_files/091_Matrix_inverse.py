import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(A)
batch = 8; N = 64
def get_inputs():
    A = torch.randn(batch, N, N)
    A = A @ A.transpose(-1, -2) + N * torch.eye(N)
    return [A]
def get_init_inputs(): return []
