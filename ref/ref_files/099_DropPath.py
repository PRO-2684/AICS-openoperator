import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device)
        output = x / keep_prob * (random_tensor >= self.drop_prob).float()
        return output
batch = 32; seq = 128; d_model = 512
def get_inputs(): return [torch.randn(batch, seq, d_model)]
def get_init_inputs(): return [0.1]
