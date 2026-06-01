import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Compute the row-wise sum
        row_sums = input_tensor.sum(dim=1, keepdim=True)

        # Avoid division by zero: only normalize rows where the sum is positive
        normalized_tensor = torch.where(
            row_sums > 0, 
            input_tensor / row_sums, 
            input_tensor
        )

        return normalized_tensor

def get_init_inputs():
    return []

def get_inputs():
    # Providing a 2D tensor for row-wise normalization
    # Shape: (batch_size, num_features)
    return [torch.randn(16, 32)]