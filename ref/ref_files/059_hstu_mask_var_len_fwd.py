import torch
import torch.nn as nn
import torch.nn.functional as F


SEQ_OFFSETS = [0, 10, 22]
Q_HEAD = 4
KV_HEAD = 2
QK_DIM = 32
V_DIM = 40
ALPHA = 0.7
MAX_SEQ_LEN = 12
CONTEXT_SIZE = [3, 2]
HISTORY_SIZE = [2, 3]
REALTIME_SIZE = [1, 1]
HISTORY_CAUSAL = [1, 0]
CONTEXT_EXPAND = [1, 0]
LABEL_LENS = [2, 2]
PREFERENCE = 2
SEED = 2026


def _normalize_preference(preference: int | str) -> str:
    if isinstance(preference, str):
        return preference
    return {
        0: "FAST",
        1: "HIGH_PRECISION",
        2: "BALANCE",
    }[int(preference)]


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    x = x.unsqueeze(2)
    x = torch.tile(x, [1, 1, n_rep, 1, 1])
    return x.reshape(x.shape[0], x.shape[1] * n_rep, x.shape[3], x.shape[4])


def _build_mask(
    q_len: int,
    k_len: int,
    ctx_size: int,
    htr_size: int,
    rlt_size: int,
    htr_causal: bool,
    ctx_expand: bool,
    lab_lens: int,
    device: torch.device,
) -> torch.Tensor:
    mask_row = torch.arange(0, q_len, device=device)
    if ctx_expand:
        mask_row[0:ctx_size] = ctx_size + htr_size - 1
    else:
        mask_row[0:ctx_size] = ctx_size - 1
    if not htr_causal:
        mask_row[ctx_size: ctx_size + htr_size] = ctx_size + htr_size - 1

    mask_col1 = torch.arange(0, k_len, device=device)
    mask_col2 = torch.ones(k_len, device=device, dtype=mask_row.dtype) * k_len
    lab_begin = ctx_size + htr_size + rlt_size
    lab_size = q_len - lab_begin
    lab_groups = lab_size // lab_lens
    for group_i in range(lab_groups):
        start = lab_begin + group_i * lab_lens
        end = lab_begin + (group_i + 1) * lab_lens
        mask_col2[start:end] = end

    mask1 = mask_row[:, None] >= mask_col1[None, :]
    mask2 = mask_row[:, None] < mask_col2[None, :]
    return mask1 & mask2


def _hstu_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    max_seq_len: int,
    ctx_size: int,
    htr_size: int,
    rlt_size: int,
    htr_causal: bool,
    ctx_expand: bool,
    lab_lens: int,
) -> torch.Tensor:
    q = torch.transpose(q, 1, 2)
    k = torch.transpose(k, 1, 2)
    v = torch.transpose(v, 1, 2)
    q_head, q_len = q.shape[1], q.shape[2]
    kv_head, k_len = k.shape[1], k.shape[2]

    if q_head > kv_head:
        head_group = q_head // kv_head
        k = _repeat_kv(k, head_group)
        v = _repeat_kv(v, head_group)

    mask = _build_mask(
        q_len=q_len,
        k_len=k_len,
        ctx_size=ctx_size,
        htr_size=htr_size,
        rlt_size=rlt_size,
        htr_causal=htr_causal,
        ctx_expand=ctx_expand,
        lab_lens=lab_lens,
        device=q.device,
    )

    logits = torch.matmul(q.float(), k.transpose(2, 3).float()) * alpha
    logits = torch.where(
        mask.unsqueeze(0).unsqueeze(0),
        logits,
        torch.full_like(logits, -1e6),
    )
    probs = F.silu(logits) * (1.0 / max_seq_len)
    probs = probs.to(q.dtype).float()
    out = torch.matmul(probs, v.float()).to(q.dtype)
    return torch.transpose(out, 1, 2)


def _hstu_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    context_size: torch.Tensor,
    history_size: torch.Tensor,
    realtime_size: torch.Tensor,
    history_causal: torch.Tensor,
    context_expand: torch.Tensor,
    label_lens: torch.Tensor,
) -> torch.Tensor:
    batch_size = seq_offsets.shape[0] - 1
    seq_lens = (seq_offsets[1:] - seq_offsets[:-1]).tolist()
    q_list = torch.split(q, seq_lens)
    k_list = torch.split(k, seq_lens)
    v_list = torch.split(v, seq_lens)

    out_list = []
    for i in range(batch_size):
        out_list.append(
            _hstu_reference(
                q_list[i].unsqueeze(0),
                k_list[i].unsqueeze(0),
                v_list[i].unsqueeze(0),
                alpha=alpha,
                max_seq_len=max_seq_len,
                ctx_size=int(context_size[i].item()),
                htr_size=int(history_size[i].item()),
                rlt_size=int(realtime_size[i].item()),
                htr_causal=bool(int(history_causal[i].item())),
                ctx_expand=bool(int(context_expand[i].item())),
                lab_lens=int(label_lens[i].item()),
            ).squeeze(0)
        )
    return torch.cat(out_list, dim=0)


class Model(nn.Module):
    """
    Reference implementation of the masked variable-length HSTU forward pass.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: float,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        context_size: torch.Tensor,
        history_size: torch.Tensor,
        realtime_size: torch.Tensor,
        history_causal: torch.Tensor,
        context_expand: torch.Tensor,
        label_lens: torch.Tensor,
        preference: int | str,
    ) -> torch.Tensor:
        """
        Computes flattened HSTU attention with context/history/realtime/label masks.
        """
        _normalize_preference(preference)
        return _hstu_varlen_reference(
            q,
            k,
            v,
            alpha,
            seq_offsets,
            max_seq_len,
            context_size,
            history_size,
            realtime_size,
            history_causal,
            context_expand,
            label_lens,
        )


def get_inputs():
    torch.manual_seed(SEED)
    total_seq_len = SEQ_OFFSETS[-1]

    q = torch.randn((total_seq_len, Q_HEAD, QK_DIM), dtype=torch.float16)
    k = torch.randn((total_seq_len, KV_HEAD, QK_DIM), dtype=torch.float16)
    v = torch.randn((total_seq_len, KV_HEAD, V_DIM), dtype=torch.float16)

    seq_offsets = torch.tensor(SEQ_OFFSETS, dtype=torch.int32)
    context_size = torch.tensor(CONTEXT_SIZE, dtype=torch.int32)
    history_size = torch.tensor(HISTORY_SIZE, dtype=torch.int32)
    realtime_size = torch.tensor(REALTIME_SIZE, dtype=torch.int32)
    history_causal = torch.tensor(HISTORY_CAUSAL, dtype=torch.int32)
    context_expand = torch.tensor(CONTEXT_EXPAND, dtype=torch.int32)
    label_lens = torch.tensor(LABEL_LENS, dtype=torch.int32)

    return [
        q,
        k,
        v,
        ALPHA,
        seq_offsets,
        MAX_SEQ_LEN,
        context_size,
        history_size,
        realtime_size,
        history_causal,
        context_expand,
        label_lens,
        PREFERENCE,
    ]


def get_init_inputs():
    return []
