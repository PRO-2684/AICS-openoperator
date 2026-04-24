import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_proj = nn.Linear(cond_dim, dim)
        self.beta_proj  = nn.Linear(cond_dim, dim)
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        gamma = self.gamma_proj(cond).unsqueeze(1)
        beta  = self.beta_proj(cond).unsqueeze(1)
        return gamma * x + beta
batch = 32; seq = 128; dim = 512; cond_dim = 256
def get_inputs(): return [torch.randn(batch, seq, dim), torch.randn(batch, cond_dim)]
def get_init_inputs(): return [dim, cond_dim]
