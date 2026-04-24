import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.emb(indices)
batch = 64; seq = 128; vocab_size = 32768; emb_dim = 512
def get_inputs(): return [torch.randint(0, vocab_size, (batch, seq))]
def get_init_inputs(): return [vocab_size, emb_dim]
