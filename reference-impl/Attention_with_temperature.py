import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__(); self.temperature = temperature
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        import math
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(d) * self.temperature)
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
batch = 2; heads = 8; seq = 128; dim = 64
def get_inputs():
    return [torch.randn(batch, heads, seq, dim)] * 3
def get_init_inputs(): return [0.5]
