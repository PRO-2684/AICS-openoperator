import torch, torch.nn as nn, math
class Model(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        return x + self.pe[:seq].unsqueeze(0)
batch = 32; seq = 128; d_model = 512
def get_inputs(): return [torch.randn(batch, seq, d_model)]
def get_init_inputs(): return [d_model]
