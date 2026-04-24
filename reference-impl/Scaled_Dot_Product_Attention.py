import torch, torch.nn as nn, torch.nn.functional as F, math
class Model(nn.Module):
    def __init__(self, scale: float = None):
        super().__init__(); self.scale = scale
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d = q.shape[-1]
        scale = self.scale if self.scale else math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
batch = 2; heads = 8; seq = 128; dim = 64
def get_inputs():
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    return [q, k, v]
def get_init_inputs(): return []
