import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu

torch.mlu = torch_mlu.mlu


class Model(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(mask == 0, float("-inf"))
        return F.softmax(x, dim=-1)


batch = 32
heads = 8
seq = 128
dtype = torch.float16


def make_inputs():
    x = torch.randn(batch, heads, seq, seq, dtype=dtype).mlu()
    mask = torch.ones(batch, 1, seq, seq, dtype=torch.bool).mlu()
    mask[:, :, :, seq // 2 :] = False
    return x, mask


def benchmark(model, warmup=20, repeats=100):
    with torch.no_grad():
        for _ in range(warmup):
            x, mask = make_inputs()
            _ = model(x, mask)
        torch.mlu.synchronize()

        start = torch.mlu.Event(enable_timing=True)
        end = torch.mlu.Event(enable_timing=True)

        start.record()
        for _ in range(repeats):
            x, mask = make_inputs()
            _ = model(x, mask)
        end.record()
        end.synchronize()

        return start.hardware_time(end) / repeats


def main():
    if not torch.mlu.is_available():
        print("MLU 不可用")
        sys.exit(1)

    torch.set_default_dtype(dtype)
    model = Model().eval().mlu()

    x, mask = make_inputs()
    with torch.no_grad():
        y = model(x, mask)
        torch.mlu.synchronize()

    avg_us = benchmark(model, warmup=20, repeats=100)

    result = {
        "device": "mlu",
        "op": "masked_fill + softmax",
        "shape": [batch, heads, seq, seq],
        "mask_shape": [batch, 1, seq, seq],
        "dtype": str(dtype),
        "output_shape": list(y.shape),
        "avg_mlu_us": avg_us,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
