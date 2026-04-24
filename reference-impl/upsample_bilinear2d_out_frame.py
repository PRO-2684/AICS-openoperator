import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, out_h: int, out_w: int, align_corners: bool = False) -> torch.Tensor:
        """
        Perform 2D bilinear upsampling on a 4D tensor (N, C, H, W).

        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W).
            out_h (int): Desired output height.
            out_w (int): Desired output width.
            align_corners (bool): Whether to align corners during interpolation.

        Returns:
            torch.Tensor: Upsampled tensor of shape (N, C, out_h, out_w).
        """
        # Extract input dimensions
        batchsize, channels, height1, width1 = input.shape

        # Compute scaling factors
        if align_corners and out_h > 1:
            rheight = (height1 - 1) / (out_h - 1)
        else:
            rheight = height1 / out_h

        if align_corners and out_w > 1:
            rwidth = (width1 - 1) / (out_w - 1)
        else:
            rwidth = width1 / out_w

        # Prepare output tensor
        output = torch.empty((batchsize, channels, out_h, out_w), device=input.device, dtype=input.dtype)

        # Iterate over output spatial dimensions
        for h2 in range(out_h):
            h1r = (h2 * rheight) if align_corners else ((h2 + 0.5) * rheight - 0.5)
            h1r = max(0, min(h1r, height1 - 1))
            h1 = int(h1r)
            h1p = min(h1 + 1, height1 - 1)
            h1lambda = h1r - h1
            h0lambda = 1.0 - h1lambda

            for w2 in range(out_w):
                w1r = (w2 * rwidth) if align_corners else ((w2 + 0.5) * rwidth - 0.5)
                w1r = max(0, min(w1r, width1 - 1))
                w1 = int(w1r)
                w1p = min(w1 + 1, width1 - 1)
                w1lambda = w1r - w1
                w0lambda = 1.0 - w1lambda

                # Perform bilinear interpolation
                for n in range(batchsize):
                    for c in range(channels):
                        v00 = input[n, c, h1, w1]
                        v01 = input[n, c, h1, w1p]
                        v10 = input[n, c, h1p, w1]
                        v11 = input[n, c, h1p, w1p]

                        output[n, c, h2, w2] = (
                            h0lambda * (w0lambda * v00 + w1lambda * v01) +
                            h1lambda * (w0lambda * v10 + w1lambda * v11)
                        )

        return output
