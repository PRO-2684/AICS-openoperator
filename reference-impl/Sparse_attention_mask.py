import torch, torch.nn as nn, torch.nn.functional as F, math
class Model(nn.Module):
    def __init__(self, window_size: int):
        super().__init__(); self.w = window_size
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq = q.shape[2]
        d = q.shape[-1]
        # Local window mask
        pos = torch.arange(seq, device=q.device)
        mask = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() <= self.w
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
batch = 2; heads = 4; seq = 256; dim = 64; window = 32
def get_inputs():
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    return [q, k, v]
def get_init_inputs(): return [32]
