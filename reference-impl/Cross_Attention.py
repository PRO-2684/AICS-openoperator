import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(query, key_value, key_value)
        return out
batch = 4; q_seq = 64; kv_seq = 128; d_model = 512; num_heads = 8
def get_inputs():
    return [torch.randn(batch, q_seq, d_model), torch.randn(batch, kv_seq, d_model)]
def get_init_inputs(): return [d_model, num_heads]
