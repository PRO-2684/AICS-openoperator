import torch, torch.nn as nn, math
class Model(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        slopes = torch.Tensor([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes)
    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=self.slopes.device)
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        bias = -self.slopes.view(-1, 1, 1) * rel_pos.unsqueeze(0)
        return bias
num_heads = 8; seq = 128
def get_inputs(): return [torch.tensor(seq)]
def get_init_inputs(): return [num_heads]
