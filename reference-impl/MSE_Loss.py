import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions, targets)
batch = 128; dim = 4096
def get_inputs(): return [torch.randn(batch, dim), torch.randn(batch, dim)]
def get_init_inputs(): return []
