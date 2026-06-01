import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(mask == 0, float('-inf'))
        return F.softmax(x, dim=-1)
batch = 32; heads = 8; seq = 128
def get_inputs():
    x = torch.randn(batch, heads, seq, seq)
    mask = torch.ones(batch, 1, seq, seq).bool()
    mask[:, :, :, seq//2:] = False
    return [x, mask]
def get_init_inputs(): return []
