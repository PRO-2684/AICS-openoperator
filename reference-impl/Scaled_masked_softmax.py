import torch, torch.nn as nn, torch.nn.functional as F, math
class Model(nn.Module):
    def __init__(self, scale: float = None):
        super().__init__(); self.scale = scale
    def forward(self, attn_weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.scale:
            attn_weight = attn_weight * self.scale
        attn_weight = attn_weight.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attn_weight, dim=-1)
batch = 4; heads = 8; seq = 128
def get_inputs():
    w = torch.randn(batch, heads, seq, seq)
    mask = torch.tril(torch.ones(seq, seq)).bool().unsqueeze(0).unsqueeze(0)
    return [w, mask]
def get_init_inputs(): return [1.0]
