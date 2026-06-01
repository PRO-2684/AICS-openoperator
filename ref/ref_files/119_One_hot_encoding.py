import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__(); self.num_classes = num_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x, self.num_classes).float()
batch = 1024; num_classes = 512
def get_inputs(): return [torch.randint(0, num_classes, (batch,))]
def get_init_inputs(): return [512]
