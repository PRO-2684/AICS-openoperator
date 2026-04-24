import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return self.loss(anchor, positive, negative)
batch = 128; dim = 256
def get_inputs():
    return [torch.randn(batch, dim), torch.randn(batch, dim), torch.randn(batch, dim)]
def get_init_inputs(): return [1.0]
