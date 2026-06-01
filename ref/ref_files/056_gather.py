import torch
import torch.nn as nn
from typing import Optional, Union


class Model(nn.Module):
    """PyTorch reference for megablocks `gather` launch."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        device = x.device
        hidden_size = x.shape[1]

        indices_long = indices.to(torch.long)
        bin_ids_long = bin_ids.to(torch.long)
        pid = torch.arange(indices.numel(), device=device, dtype=torch.int32)
        bin_starts = torch.cat([bins.new_zeros(1), bins[:-1]], dim=0)

        # gather launch is equivalent to padded_gather with padded_bins == bins
        offset_in_bin = pid - bin_starts[bin_ids_long]
        index_b = offset_in_bin + bin_starts[bin_ids_long]

        # A_TO_B=True path:
        # source row = index_a // top_k
        src_rows = torch.div(indices, top_k, rounding_mode="floor").to(torch.long)
        values = torch.index_select(x, dim=0, index=src_rows)

        if weights is not None:
            scales = torch.index_select(weights, dim=0, index=indices_long).unsqueeze(1)
            values = values * scales

        out = x.new_zeros((int(x.shape[0] * top_k), hidden_size))
        out[index_b.to(torch.long)] = values
        return out


tokens = 4
hidden_size = 8
top_k = 2
indices_data = [0, 3, 5, 1, 6, 2, 4, 7]
bin_ids_data = [0, 0, 0, 1, 1, 2, 2, 2]
weights_data = [1.00, 0.50, 0.80, 0.75, 1.10, 1.60, 1.25, 0.90]
bins_data = [3, 5, 8]


def get_inputs():
    """
    Returns a valid routed test case for `gather`.

    case0 shapes:
        x       : (4, 8)
        indices : (8,)
        bin_ids : (8,)
        weights : (8,)
        bins    : (3,)
        top_k   : scalar

    gather output shape:
        (tokens * top_k, hidden_size) = (8, 8)
    """
    x = torch.arange(tokens * hidden_size, dtype=torch.float16).view(tokens, hidden_size)
    indices = torch.tensor(indices_data, dtype=torch.int32)
    bin_ids = torch.tensor(bin_ids_data, dtype=torch.int32)
    weights = torch.tensor(weights_data, dtype=torch.float16)
    bins = torch.tensor(bins_data, dtype=torch.int32)

    return [x, indices, bin_ids, weights, bins, top_k]


def get_init_inputs():
    return []
