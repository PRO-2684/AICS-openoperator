import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x: torch.Tensor):
        output, (h_n, c_n) = self.lstm(x)
        return output
batch = 32; seq = 128; input_size = 256; hidden_size = 512; num_layers = 2
def get_inputs(): return [torch.randn(batch, seq, input_size)]
def get_init_inputs(): return [input_size, hidden_size, num_layers]
