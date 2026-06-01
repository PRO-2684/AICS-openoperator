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

batch_size = 2
channels = 4
in_T, in_H, in_W = 16, 16, 16
out_T, out_H, out_W = 8, 8, 8

def get_inputs():
    input_tensor = torch.randn(batch_size, channels, in_T, in_H, in_W)
    return [input_tensor, out_T, out_H, out_W]

def get_init_inputs():
    return []