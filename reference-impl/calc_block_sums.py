import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.Tensor, iters_per_cta: int):
        """
        Compute block-wise sums of the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (N,).
            iters_per_cta (int): Number of iterations per block.

        Returns:
            torch.Tensor: Tensor of block-wise sums.
        """
        BLOCK_THREADS = 128
        ITEMS_PER_THREAD = 4

        # Calculate the number of elements per block
        elems_per_block = BLOCK_THREADS * ITEMS_PER_THREAD * iters_per_cta

        # Calculate the number of blocks
        num_blocks = (input.numel() + elems_per_block - 1) // elems_per_block

        # Pad the input tensor to make it divisible by elems_per_block
        pad_size = elems_per_block * num_blocks - input.numel()
        if pad_size > 0:
            input = torch.nn.functional.pad(input, (0, pad_size), mode='constant', value=0)

        # Reshape the input tensor into blocks
        input_blocks = input.view(num_blocks, elems_per_block)

        # Compute the sum of each block using efficient vectorized operations
        block_sums = torch.sum(input_blocks, dim=1)

        return block_sums
