import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, osizeT: int, osizeH: int, osizeW: int) -> torch.Tensor:
        """
        Perform 3D adaptive average pooling on a 5D input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C, T, H, W).
            osizeT (int): Target size for the temporal dimension.
            osizeH (int): Target size for the height dimension.
            osizeW (int): Target size for the width dimension.

        Returns:
            torch.Tensor: Output tensor of shape (N, C, osizeT, osizeH, osizeW).
        """
        # Use PyTorch's built-in adaptive_avg_pool3d for efficient computation
        return F.adaptive_avg_pool3d(input, (osizeT, osizeH, osizeW))
