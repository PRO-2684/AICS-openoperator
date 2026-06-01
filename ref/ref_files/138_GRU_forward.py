import torch, torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        weight_ih_l0: torch.Tensor,
        weight_hh_l0: torch.Tensor,
        bias_ih_l0: torch.Tensor,
        bias_hh_l0: torch.Tensor,
        weight_ih_l1: torch.Tensor,
        weight_hh_l1: torch.Tensor,
        bias_ih_l1: torch.Tensor,
        bias_hh_l1: torch.Tensor,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru.weight_ih_l0.data.copy_(weight_ih_l0)
        self.gru.weight_hh_l0.data.copy_(weight_hh_l0)
        self.gru.bias_ih_l0.data.copy_(bias_ih_l0)
        self.gru.bias_hh_l0.data.copy_(bias_hh_l0)
        self.gru.weight_ih_l1.data.copy_(weight_ih_l1)
        self.gru.weight_hh_l1.data.copy_(weight_hh_l1)
        self.gru.bias_ih_l1.data.copy_(bias_ih_l1)
        self.gru.bias_hh_l1.data.copy_(bias_hh_l1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output


batch = 32
seq = 128
input_size = 256
hidden_size = 512
num_layers = 2


def get_inputs():
    return [torch.randn(batch, seq, input_size)]


def get_init_inputs():
    weight_ih_l0 = torch.randn(3 * hidden_size, input_size)
    weight_hh_l0 = torch.randn(3 * hidden_size, hidden_size)
    bias_ih_l0 = torch.randn(3 * hidden_size)
    bias_hh_l0 = torch.randn(3 * hidden_size)
    weight_ih_l1 = torch.randn(3 * hidden_size, hidden_size)
    weight_hh_l1 = torch.randn(3 * hidden_size, hidden_size)
    bias_ih_l1 = torch.randn(3 * hidden_size)
    bias_hh_l1 = torch.randn(3 * hidden_size)
    return [
        weight_ih_l0,
        weight_hh_l0,
        bias_ih_l0,
        bias_hh_l0,
        weight_ih_l1,
        weight_hh_l1,
        bias_ih_l1,
        bias_hh_l1,
        input_size,
        hidden_size,
        num_layers,
    ]
