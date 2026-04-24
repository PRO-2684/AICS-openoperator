import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(input, 1, index)
batch = 64; N = 1024; K = 32
def get_inputs():
    inp = torch.randn(batch, N)
    idx = torch.randint(0, N, (batch, K))
    return [inp, idx]
def get_init_inputs(): return []
