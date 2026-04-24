import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
batch = 64; seq = 128; in_features = 512; out_features = 2048
def get_inputs(): return [torch.randn(batch, seq, in_features)]
def get_init_inputs(): return [in_features, out_features]
