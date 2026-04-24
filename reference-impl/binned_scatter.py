from typing import Optional, Union

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Performs an expert-binned scatter-reduce operation.

    Args:
        No initialization arguments are required.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Scatters expert-binned rows back to token rows and reduces across top-k routes.

        Args:
            x (torch.Tensor): Input tensor of shape (num_experts, expert_capacity, hidden_size).
            indices (torch.Tensor): Routed indices of shape (tokens * top_k,).
            weights (Optional[torch.Tensor]): Optional per-route scaling factors of shape (tokens * top_k,).
            bins (torch.Tensor): Cumulative routed counts for each expert, shape (num_experts,).
            top_k (Union[int, torch.Tensor]): Number of routed entries per source token.

        Returns:
            torch.Tensor: Output tensor of shape (tokens, hidden_size).
        """
        top_k = int(top_k.item()) if torch.is_tensor(top_k) else int(top_k)

        num_experts = x.shape[0]
        tokens = indices.shape[0] // top_k
        hidden_size = x.shape[2]
        out = x.new_zeros((tokens, hidden_size))

        start = 0
        for expert_idx in range(num_experts):
            end = int(bins[expert_idx].item())
            num_tokens = end - start
            expert_capacity = x.shape[1]
            for entry_idx in range(min(expert_capacity, num_tokens)):
                index_a = int(indices[start + entry_idx].item())
                values = x[expert_idx, entry_idx]
                if weights is not None:
                    values = values * weights[index_a]
                out[index_a // top_k] += values
            start = end

        return out


num_experts = 2
expert_capacity = 5
hidden_size = 4
top_k = 2
indices_data = [0, 2, 4, 6, 1, 3, 5, 7]
weights_data = [1.0, 0.5, 1.25, 0.75, 1.5, 0.8, 1.1, 0.9]
bins_data = [4, 8]


def get_inputs():
    x = torch.arange(
        num_experts * expert_capacity * hidden_size,
        dtype=torch.float16,
    ).view(num_experts, expert_capacity, hidden_size)
    indices = torch.tensor(indices_data, dtype=torch.int32)
    weights = torch.tensor(weights_data, dtype=torch.float16)
    bins = torch.tensor(bins_data, dtype=torch.int32)
    return [x, indices, weights, bins, top_k]


def get_init_inputs():
    return []
