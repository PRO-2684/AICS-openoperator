from typing import Optional, Union

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Performs an expert-binned gather operation.

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
        expert_capacity: Union[int, torch.Tensor],
        top_k: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Gathers routed token rows into per-expert capacity slots.

        Args:
            x (torch.Tensor): Input tensor of shape (tokens, hidden_size).
            indices (torch.Tensor): Routed indices of shape (tokens * top_k,).
            weights (Optional[torch.Tensor]): Optional per-route scaling factors of shape (tokens * top_k,).
            bins (torch.Tensor): Cumulative routed counts for each expert, shape (num_experts,).
            expert_capacity (Union[int, torch.Tensor]): Capacity allocated for each expert in the output.
            top_k (Union[int, torch.Tensor]): Replication factor used to map routed indices back to source rows.

        Returns:
            torch.Tensor: Output tensor of shape (num_experts, expert_capacity, hidden_size).
        """
        expert_capacity = (
            int(expert_capacity.item()) if torch.is_tensor(expert_capacity) else int(expert_capacity)
        )
        top_k = int(top_k.item()) if torch.is_tensor(top_k) else int(top_k)

        num_experts = bins.shape[0]
        hidden_size = x.shape[1]
        out = x.new_zeros((num_experts, expert_capacity, hidden_size))

        start = 0
        for expert_idx in range(num_experts):
            end = int(bins[expert_idx].item())
            num_tokens = end - start
            for entry_idx in range(min(expert_capacity, num_tokens)):
                index_a = int(indices[start + entry_idx].item())
                src_row = index_a // top_k
                values = x[src_row]
                if weights is not None:
                    values = values * weights[index_a]
                out[expert_idx, entry_idx] = values
            start = end

        return out


tokens = 4
hidden_size = 4
top_k = 2
expert_capacity = 5
indices_data = [0, 2, 4, 6, 1, 3, 5, 7]
weights_data = [1.0, 0.5, 1.25, 0.75, 1.5, 0.8, 1.1, 0.9]
bins_data = [4, 8]


def get_inputs():
    x = torch.arange(tokens * hidden_size, dtype=torch.float16).view(tokens, hidden_size)
    indices = torch.tensor(indices_data, dtype=torch.int32)
    weights = torch.tensor(weights_data, dtype=torch.float16)
    bins = torch.tensor(bins_data, dtype=torch.int32)
    return [x, indices, weights, bins, expert_capacity, top_k]


def get_init_inputs():
    return []
