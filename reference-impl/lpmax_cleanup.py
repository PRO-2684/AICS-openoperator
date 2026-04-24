import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_per_tensor: torch.Tensor, max_chunks_per_tensor: int):
        """
        Computes the maximum value for each row in the 2D tensor `output_per_tensor`.
        If a row contains NaN values, the result for that row will also be NaN.

        Args:
            output_per_tensor (torch.Tensor): A 2D tensor of shape [num_tensors, max_chunks_per_tensor].
            max_chunks_per_tensor (int): The number of chunks per tensor (must match output_per_tensor.size(1)).

        Returns:
            torch.Tensor: A 1D tensor of shape [num_tensors] containing the maximum value for each row.
        """
        # Ensure the input tensor is 2D and the second dimension matches max_chunks_per_tensor
        assert output_per_tensor.dim() == 2, "output_per_tensor must be a 2D tensor"
        assert output_per_tensor.size(1) == max_chunks_per_tensor, (
            "max_chunks_per_tensor must match output_per_tensor.size(1)"
        )

        # Replace NaN values with -inf to ensure they don't affect the max computation
        nan_mask = torch.isnan(output_per_tensor)
        output_no_nan = output_per_tensor.clone()
        output_no_nan[nan_mask] = float('-inf')

        # Compute the maximum value for each row
        row_max, _ = torch.max(output_no_nan, dim=1)

        # If a row contains all NaNs, the result should be NaN
        all_nan_mask = nan_mask.all(dim=1)
        row_max[all_nan_mask] = float('nan')

        return row_max
