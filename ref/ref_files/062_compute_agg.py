import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        """
        Perform an inclusive prefix sum (scan) on the input tensor `agg`.

        Args:
            agg (torch.Tensor): A 1D tensor of dtype torch.int32.

        Returns:
            torch.Tensor: A 1D tensor of dtype torch.int64 containing the inclusive prefix sum.
        """
        # Ensure the input tensor is of dtype int32
        assert agg.dtype == torch.int32, "Input tensor must be of dtype torch.int32"
        # Perform inclusive prefix sum using torch.cumsum and cast to int64
        return torch.cumsum(agg, dim=0, dtype=torch.int64)

def get_init_inputs():
    return []

def get_inputs():
    # Example input: a 1D tensor of length 1024 with random int32 values
    size = 1024
    return [torch.randint(-1000, 1000, (size,), dtype=torch.int32)]