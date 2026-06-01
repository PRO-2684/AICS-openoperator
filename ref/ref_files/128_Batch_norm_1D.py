import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
batch = 256; features = 512
def get_inputs(): return [torch.randn(batch, features)]
def get_init_inputs(): return [512]
