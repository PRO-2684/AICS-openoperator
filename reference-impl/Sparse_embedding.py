import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, sparse=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)
batch = 256; seq = 64; vocab = 50000; dim = 256
def get_inputs(): return [torch.randint(0, vocab, (batch, seq))]
def get_init_inputs(): return [50000, 256]
