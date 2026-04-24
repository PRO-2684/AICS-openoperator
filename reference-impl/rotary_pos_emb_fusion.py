import torch
import torch.nn as nn


BATCH_SIZE = 2
SEQ_LEN = 64
NUM_HEADS = 4
HEAD_DIM = 64
UNSQUEEZE_DIM = 1
SEED = 0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


class Model(nn.Module):
    """
    Apply rotary positional embeddings to Q and K tensors.

    Args:
        unsqueeze_dim: Broadcast dimension used when expanding cos/sin.
    """

    def __init__(self, unsqueeze_dim: int = 1):
        super().__init__()
        self.unsqueeze_dim = unsqueeze_dim

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        """
        Apply RoPE to Q and K tensors with shape [batch, seq_len, num_heads, head_dim].

        Args:
            q: Query tensor.
            k: Key tensor.
            cos: Cosine cache with shape [seq_len, head_dim].
            sin: Sine cache with shape [seq_len, head_dim].

        Returns:
            A tuple `(q_embed, k_embed)` with the same shapes as `q` and `k`.
        """
        cos = cos.unsqueeze(self.unsqueeze_dim)
        sin = sin.unsqueeze(self.unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


def get_inputs():
    torch.manual_seed(SEED)
    q = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    cos = torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float16)
    sin = torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float16)
    return [q, k, cos, sin]


def get_init_inputs():
    return [UNSQUEEZE_DIM]
