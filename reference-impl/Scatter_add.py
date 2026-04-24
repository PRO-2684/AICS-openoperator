import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
        return out
N = 4096; D = 256; dim_size = 512
def get_inputs():
    src = torch.randn(N, D)
    index = torch.randint(0, dim_size, (N,))
    return [src, index, dim_size]
def get_init_inputs(): return []
