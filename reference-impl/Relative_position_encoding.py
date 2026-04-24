import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, max_seq: int, d_model: int):
        super().__init__()
        self.rel_emb = nn.Embedding(2 * max_seq - 1, d_model)
        self.max_seq = max_seq
    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1) + self.max_seq - 1
        return self.rel_emb(rel)
max_seq = 256; d_model = 64; seq = 128
def get_inputs(): return [torch.tensor(seq)]
def get_init_inputs(): return [max_seq, d_model]
