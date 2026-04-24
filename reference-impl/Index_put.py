import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, input: torch.Tensor, values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        out = input.clone()
        out[index] = values
        return out
M = 4096; N = 512; K = 256
def get_inputs():
    inp = torch.randn(M, N)
    vals = torch.randn(K, N)
    idx = torch.randint(0, M, (K,)).unique()
    idx = idx[:K] if len(idx) >= K else idx
    vals = vals[:len(idx)]
    return [inp, vals, idx]
def get_init_inputs(): return []
