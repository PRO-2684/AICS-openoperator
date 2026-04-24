import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(A, b, upper=True)
batch = 8; N = 64
def get_inputs():
    A = torch.triu(torch.randn(batch, N, N)) + N * torch.eye(N)
    b = torch.randn(batch, N, 1)
    return [A, b]
def get_init_inputs(): return []
