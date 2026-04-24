import torch, torch.nn as nn, torch.nn.functional as F, math
class Model(nn.Module):
    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                k_new: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
        k = torch.cat([k_cache, k_new], dim=2)
        v = torch.cat([v_cache, v_new], dim=2)
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
batch = 1; heads = 8; cache_len = 512; new_len = 1; dim = 64
def get_inputs():
    q  = torch.randn(batch, heads, new_len, dim)
    kc = torch.randn(batch, heads, cache_len, dim)
    vc = torch.randn(batch, heads, cache_len, dim)
    kn = torch.randn(batch, heads, new_len, dim)
    vn = torch.randn(batch, heads, new_len, dim)
    return [q, kc, vc, kn, vn]
def get_init_inputs(): return []
