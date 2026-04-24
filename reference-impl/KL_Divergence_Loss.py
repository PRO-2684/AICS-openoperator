import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def forward(self, input_log_prob: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        return F.kl_div(input_log_prob, target_prob, reduction='batchmean')
batch = 128; num_classes = 1024
def get_inputs():
    p = F.softmax(torch.randn(batch, num_classes), dim=-1)
    q = F.softmax(torch.randn(batch, num_classes), dim=-1)
    return [torch.log(p), q]
def get_init_inputs(): return []
