import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)
batch = 32; seq = 128; d_model = 512
def get_inputs(): return [torch.randn(batch, seq, d_model)]
def get_init_inputs(): return [d_model]
