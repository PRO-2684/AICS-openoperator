import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, osizeH: int, osizeW: int) -> torch.Tensor:
        """
        Perform adaptive average pooling on an NHWC tensor using PyTorch's built-in operators.

        Args:
            input (torch.Tensor): Input tensor of shape (B, H, W, C) in NHWC format.
            osizeH (int): Target output height.
            osizeW (int): Target output width.

        Returns:
            torch.Tensor: Output tensor of shape (B, osizeH, osizeW, C) in NHWC format.
        """
        # Convert NHWC to NCHW for PyTorch's adaptive_avg_pool2d
        input_nchw = input.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Apply adaptive average pooling
        output_nchw = F.adaptive_avg_pool2d(input_nchw, (osizeH, osizeW))  # (B, C, osizeH, osizeW)

        # Convert back to NHWC format
        output_nhwc = output_nchw.permute(0, 2, 3, 1)  # (B, osizeH, osizeW, C)

        return output_nhwc

batch_size = 8
in_height = 32
in_width = 32
in_channels = 16
target_height = 16
target_width = 16

def get_inputs():
    # input: (B, H, W, C), osizeH: int, osizeW: int
    input_tensor = torch.randn(batch_size, in_height, in_width, in_channels)
    return [input_tensor, target_height, target_width]

def get_init_inputs():
    return []