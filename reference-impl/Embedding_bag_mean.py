import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.emb_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='mean')
    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.emb_bag(indices, offsets)
num_bags = 64; avg_len = 32; vocab_size = 32768; emb_dim = 256
def get_inputs():
    indices = torch.randint(0, vocab_size, (num_bags * avg_len,))
    offsets = torch.arange(0, num_bags * avg_len, avg_len)
    return [indices, offsets]
def get_init_inputs(): return [vocab_size, emb_dim]
