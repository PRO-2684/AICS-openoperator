import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference model for the Distributed Cross Entropy kernel with Label Smoothing.
    This model simulates the behavior of the Triton kernel, including handling 
    of the ignore_index and the normalization by non-ignored elements.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, _input, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx):
        label_smoothing = float(label_smoothing.item()) if isinstance(label_smoothing, torch.Tensor) else label_smoothing
        reduce_loss = bool(reduce_loss.item())
        ignore_idx = int(ignore_idx.item())
        # 1. 硬件兼容性处理：强制转为 float16
        # 解决 "nll_loss_forward... not implemented for BFloat16" 
        orig_dtype = _input.dtype
        logits = _input.to(torch.float16).view(-1, _input.size(-1))
        target = target.to(torch.long).view(-1)

        # 2. 调用 PyTorch 参考实现
        loss = F.cross_entropy(
            logits, 
            target, 
            ignore_index=ignore_idx, 
            label_smoothing=label_smoothing, 
            reduction='none'
        )

        # 3. 按照 Triton 算子的逻辑进行归约
        if reduce_loss:
            n_non_ignore = (target != ignore_idx).sum().to(torch.float16)
            # 强制转为 1 维
            loss = (loss.sum() / n_non_ignore.clamp(min=1.0)).view(1)
        else:
            loss = loss.view(_input.shape[0], _input.shape[1])

        return loss.to(orig_dtype)
        
# 测试参数设置
batch_size = 4
seq_len = 128
vocab_size = 32768

# 定义控制参数
label_smoothing = 0.1
reduce_loss = True
dist_process_group = None  # 测试环境通常不涉及分布式组
ignore_idx = -100

def get_inputs():
    """
    Generates representative inputs for the cross entropy kernel.
    """
    ls = torch.tensor(label_smoothing)
    rl = torch.tensor(1) # 1 为 True, 0 为 False
    idx = torch.tensor(-100)
    dist_process_group = torch.tensor([])

    # 模拟 Logits (B, SQ, V)
    _input = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float16)
    
    # 模拟标签 (B * SQ)，随机包含一些 ignore_idx
    target = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # # 随机插入 ignore_idx 以测试过滤逻辑
    # mask = torch.rand(target.shape) < 0.1
    # target[mask] = idx
    # 将所有参数打包返回
    # 注意：顺序必须与 Model.forward 的参数顺序完全一致
    return [_input, target, ls, rl, dist_process_group, idx]

def get_init_inputs():
    return []
