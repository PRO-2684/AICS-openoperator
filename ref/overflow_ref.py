import torch
import torch.nn.functional as F

import torch_mlu

torch.mlu = torch_mlu.mlu  # type: ignore[attr-defined]

batch_size = 4
seq_len = 128
vocab_size = 32768

label_smoothing = 0.1
ignore_idx = -100
trials = 2000

device = "mlu"


@torch.no_grad()
def ref_chain_once():
    # 模拟测评精度阶段：raw fp16 -> _move_nested_tensors_to_mlu_fp32 -> fp32 MLU
    _input = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        dtype=torch.float16,
        device=device,
    ).to(torch.float32)

    target = torch.randint(
        0,
        vocab_size,
        (batch_size * seq_len,),
        device=device,
    )

    orig_dtype = _input.dtype

    # 你要测试的关键链条
    logits = _input.to(torch.float16).view(-1, _input.size(-1))
    target = target.to(torch.long).view(-1)

    loss_vec = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_idx,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    n_non_ignore = (target != ignore_idx).sum().to(torch.float16)
    out = (loss_vec.sum() / n_non_ignore.clamp(min=1.0)).view(1)

    out = out.to(orig_dtype)

    torch.mlu.synchronize()

    loss_cpu = loss_vec.detach().cpu()
    out_cpu = out.detach().cpu()

    return {
        "out": float(out_cpu.item()),
        "out_finite": bool(torch.isfinite(out_cpu).all().item()),
        "out_inf": bool(torch.isinf(out_cpu).any().item()),
        "out_nan": bool(torch.isnan(out_cpu).any().item()),
        "loss_total": loss_cpu.numel(),
        "loss_finite": int(torch.isfinite(loss_cpu).sum().item()),
        "loss_inf": int(torch.isinf(loss_cpu).sum().item()),
        "loss_nan": int(torch.isnan(loss_cpu).sum().item()),
    }


def main():
    assert torch.mlu.is_available(), "MLU not available"

    # warmup
    for _ in range(10):
        ref_chain_once()

    out_finite = 0
    out_inf = 0
    out_nan = 0

    loss_total = 0
    loss_inf = 0
    loss_nan = 0
    loss_all_inf_trials = 0
    loss_any_inf_trials = 0

    examples = []

    for i in range(trials):
        r = ref_chain_once()

        out_finite += int(r["out_finite"])
        out_inf += int(r["out_inf"])
        out_nan += int(r["out_nan"])

        loss_total += r["loss_total"]
        loss_inf += r["loss_inf"]
        loss_nan += r["loss_nan"]

        loss_any_inf_trials += int(r["loss_inf"] > 0)
        loss_all_inf_trials += int(r["loss_inf"] == r["loss_total"])

        if len(examples) < 10:
            examples.append(r)

        if (i + 1) % 100 == 0:
            print(f"done {i + 1}/{trials}")

    print()
    print("==== ref chain: fp32 input -> fp16 logits -> CE -> fp32 output ====")
    print(f"trials                  = {trials}")
    print(f"out finite count        = {out_finite}/{trials}")
    print(f"out finite rate         = {out_finite / trials * 100:.4f}%")
    print(f"out inf count           = {out_inf}/{trials}")
    print(f"out inf rate            = {out_inf / trials * 100:.4f}%")
    print(f"out nan count           = {out_nan}/{trials}")
    print(f"out nan rate            = {out_nan / trials * 100:.4f}%")

    print()
    print("==== loss vector stats ====")
    print(f"loss total elems        = {loss_total}")
    print(f"loss inf elems          = {loss_inf}")
    print(f"loss inf elem rate      = {loss_inf / loss_total * 100:.4f}%")
    print(f"loss nan elems          = {loss_nan}")
    print(f"loss nan elem rate      = {loss_nan / loss_total * 100:.4f}%")
    print(f"trials any loss inf     = {loss_any_inf_trials}/{trials}")
    print(f"trials any loss inf rate= {loss_any_inf_trials / trials * 100:.4f}%")
    print(f"trials all loss inf     = {loss_all_inf_trials}/{trials}")
    print(f"trials all loss inf rate= {loss_all_inf_trials / trials * 100:.4f}%")

    print()
    print("==== first examples ====")
    for e in examples:
        print(e)


if __name__ == "__main__":
    main()
