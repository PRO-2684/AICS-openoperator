import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor):
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U, S, Vh
batch = 8; M = 128; N = 64
def get_inputs(): return [torch.randn(batch, M, N)]
def get_init_inputs(): return []
