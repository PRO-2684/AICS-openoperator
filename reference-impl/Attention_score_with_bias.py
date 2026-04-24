import torch, torch.nn as nn, torch.nn.functional as F, math
class Model(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d) + bias
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
batch = 2; heads = 8; seq = 128; dim = 64
def get_inputs():
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    bias = torch.randn(1, heads, seq, seq)
    return [q, k, v, bias]
def get_init_inputs(): return []
