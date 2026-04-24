import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        bias1: torch.Tensor,
        bias2: torch.Tensor,
        cx: torch.Tensor,
        hsz: int,
        totalElements: int,
    ) -> torch.Tensor:
        # Match original wrapper: return only hy of shape (totalElements,)

        # Ensure contiguity like in C++ wrapper
        input_c = input.contiguous()
        hidden_c = hidden.contiguous()
        bias1_c = bias1.contiguous()
        bias2_c = bias2.contiguous()
        cx_c = cx.contiguous()

        batch = totalElements // hsz

        # Reshape to (batch, 4, hsz) to mimic CUDA layout [batch, 4, hsz]
        input_reshaped = input_c.view(batch, 4, hsz)
        hidden_reshaped = hidden_c.view(batch, 4, hsz)

        # Stack input and hidden, then sum to avoid per-gate indexing
        # gates: (batch, 4, hsz)
        gates = input_reshaped + hidden_reshaped

        # Biases: (4, hsz) -> broadcast to (1, 4, hsz)
        bias_sum = (bias1_c + bias2_c).unsqueeze(0)

        gates = gates + bias_sum

        # Split gates along gate dimension (dim=1)
        ig, fg, cg, og = gates.unbind(dim=1)

        ig = torch.sigmoid(ig)
        fg = torch.sigmoid(fg)
        cg = torch.tanh(cg)
        og = torch.sigmoid(og)

        cx_reshaped = cx_c.view(batch, hsz)

        cy = fg * cx_reshaped + ig * cg
        hy = og * torch.tanh(cy)

        return hy.reshape(totalElements)
