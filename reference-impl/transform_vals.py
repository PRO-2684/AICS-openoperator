import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Ensure the inputs are of the same shape and type
        assert a.shape == b.shape, "Input tensors must have the same shape"
        assert a.dtype == torch.float16 and b.dtype == torch.float16, "Input tensors must be of type float16"
        
        # Perform elementwise addition
        return a + b
