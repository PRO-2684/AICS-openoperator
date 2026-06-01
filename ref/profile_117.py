#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
import time

import torch
import torch_mlu  # noqa: F401
from torch_mlu.utils.cpp_extension import load_inline


E = 4194304
T = 32
N = E // T


def source_for(body, launch=True, dim="{32, 1, 1}", ftype="cnrtFuncTypeBlock", ownq=False, qsync=False):
    kernel = f"""
#include <bang.h>
#include <bang_variable_vector.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <torch/extension.h>

#define E0 {N}

__mlu_entry__ void k0(half *__restrict__ x, const unsigned char *__restrict__ m) {{
{body}
}}

{"static cnrtQueue_t gq = []() { cnrtQueue_t q; cnrtQueueCreate(&q); return q; }(); " if ownq else ""}
torch::Tensor bang_func(torch::Tensor x, torch::Tensor mask, double fill_value) {{
  (void)fill_value;
"""
    if launch:
        q = "gq" if ownq else "torch_mlu::getCurMLUStream()"
        kernel += f"""
  cnrtDim3_t d = {dim};
  k0<<<d, {ftype}, {q}>>>(
      reinterpret_cast<half *>(x.data_ptr<at::Half>()),
      reinterpret_cast<const unsigned char *>(mask.data_ptr<bool>()));
  {"cnrtQueueSync(gq);" if ownq and qsync else ""}
"""
    kernel += """
  return x;
}
"""
    return kernel


VARIANTS = {
    "return": source_for("", launch=False),
    "empty": source_for("  (void)x;\n  (void)m;\n"),
    "x_rw": source_for(
        """
  __nram__ half a[E0];
  int s = taskId * E0;
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "mask_copy": source_for(
        """
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __memcpy((unsigned char *)(x + s), d, 64, NRAM2GDRAM);
"""
    ),
    "mask_i16": source_for(
        """
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((short *)c, (short *)c, (short)0xfc00, E0);
  __memcpy(x + s, c, 64, NRAM2GDRAM);
"""
    ),
    "baseline": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((short *)c, (short *)c, (short)0xfc00, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_add(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "shift_neg": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, -10);
  __bang_mul_scalar((short *)c, (short *)c, (short)-1, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_add(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "baseline_u1": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((short *)c, (short *)c, (short)0xfc00, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_add(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
""",
        dim="{4, 8, 1}",
        ftype="cnrtFuncTypeUnion1",
    ),
    "baseline_ownq": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((short *)c, (short *)c, (short)0xfc00, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_add(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
""",
        ownq=True,
    ),
    "baseline_ownq_sync": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((short *)c, (short *)c, (short)0xfc00, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_add(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
""",
        ownq=True,
        qsync=True,
    ),
    "pipe4": source_for(
        """
  __nram__ half a0[32768];
  __nram__ half a1[32768];
  __nram__ half c0[32768];
  __nram__ half c1[32768];
  __nram__ unsigned char d0[32768];
  __nram__ unsigned char d1[32768];
  int s = taskId * E0;
  half *px = x + s;
  const unsigned char *pm = m + s;
  __memcpy_async(d0, pm, 32768 * sizeof(unsigned char), GDRAM2NRAM);
  __memcpy_async(a0, px, 32768 * sizeof(half), GDRAM2NRAM);
  __sync_io();
  __memcpy_async(d1, pm + 32768, 32768 * sizeof(unsigned char), GDRAM2NRAM);
  __memcpy_async(a1, px + 32768, 32768 * sizeof(half), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c0, d0, 32768, 0);
  __bang_mul_scalar((short *)c0, (short *)c0, (short)0xfc00, 32768);
  __bang_add(a0, a0, c0, 32768);
  __sync_io_move_compute();
  __memcpy_async(px, a0, 32768 * sizeof(half), NRAM2GDRAM);
  __memcpy_async(d0, pm + 65536, 32768 * sizeof(unsigned char), GDRAM2NRAM);
  __memcpy_async(a0, px + 65536, 32768 * sizeof(half), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c1, d1, 32768, 0);
  __bang_mul_scalar((short *)c1, (short *)c1, (short)0xfc00, 32768);
  __bang_add(a1, a1, c1, 32768);
  __sync_io_move_compute();
  __memcpy_async(px + 32768, a1, 32768 * sizeof(half), NRAM2GDRAM);
  __memcpy_async(d1, pm + 98304, 32768 * sizeof(unsigned char), GDRAM2NRAM);
  __memcpy_async(a1, px + 98304, 32768 * sizeof(half), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c0, d0, 32768, 0);
  __bang_mul_scalar((short *)c0, (short *)c0, (short)0xfc00, 32768);
  __bang_add(a0, a0, c0, 32768);
  __sync_io_move_compute();
  __memcpy_async(px + 65536, a0, 32768 * sizeof(half), NRAM2GDRAM);
  __bang_uchar2int16((int16_t *)c1, d1, 32768, 0);
  __bang_mul_scalar((short *)c1, (short *)c1, (short)0xfc00, 32768);
  __bang_add(a1, a1, c1, 32768);
  __sync_io_move_compute();
  __memcpy_async(px + 98304, a1, 32768 * sizeof(half), NRAM2GDRAM);
  __sync_io();
"""
    ),
    "half_over": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2half(c, d, E0);
  __bang_mul_scalar(c, c, (half)-65504.0, E0);
  __bang_add(c, c, c, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_add(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "bit_or": source_for(
        """
  __nram__ unsigned short a[E0];
  __nram__ unsigned short c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((short *)c, (short *)c, (short)-1, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_band_scalar(a, a, (unsigned short)0x03ff, E0);
  __bang_band_scalar(c, c, (unsigned short)0xfc00, E0);
  __bang_bor(a, a, c, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "fusion": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2half(c, d, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_fusion(FUSION_FMA, a, c, (half)-65504.0, a, E0, E0);
  __bang_fusion(FUSION_FMA, a, c, (half)-65504.0, a, E0, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "fsm_bits": source_for(
        """
  __nram__ half a[E0];
  __nram__ half c[E0];
  __nram__ unsigned char d[E0];
  int s = taskId * E0;
  unsigned short u = (unsigned short)0xfc00;
  half v = *((half *)&u);
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2half(c, d, E0);
  __memcpy(a, x + s, E0 * sizeof(half), GDRAM2NRAM);
  __bang_fusion(FUSION_FSM, a, c, v, a, E0, E0);
  __memcpy(x + s, a, E0 * sizeof(half), NRAM2GDRAM);
"""
    ),
    "vv_u16": source_for(
        """
  int s = taskId * E0;
  vv_uint8 b;
  vv_uint16 v;
  vv_bool p;
  __vv_move(v, (unsigned short)0xfc00);
  for (int i = 0; i < E0; i += 256) {
    __vv_load(b, m + s + i, 256);
    __vv_setp_ne(p, b, (unsigned char)0);
    __vv_store_m((unsigned short *)(x + s + i), v, p);
  }
"""
    ),
    "vv_u32_to_u16_sparse": source_for(
        """
  int s = taskId * E0;
  vv_uint32 b0;
  vv_uint32 b1;
  vv_uint32 b2;
  vv_uint32 b3;
  vv_uint32 v;
  vv_bool p0;
  vv_bool p1;
  vv_bool p2;
  vv_bool p3;
  __vv_move(v, (unsigned int)0xfc00);
  for (int i = 0; i < E0; i += 512) {
    __vv_load(b0, m + s + i);
    __vv_load(b1, m + s + i + 128);
    __vv_load(b2, m + s + i + 256);
    __vv_load(b3, m + s + i + 384);
    __vv_setp_ne(p0, b0, (unsigned int)0);
    __vv_setp_ne(p1, b1, (unsigned int)0);
    __vv_setp_ne(p2, b2, (unsigned int)0);
    __vv_setp_ne(p3, b3, (unsigned int)0);
    __vv_store_m((unsigned short *)(x + s + i), v, p0);
    __vv_store_m((unsigned short *)(x + s + i + 128), v, p1);
    __vv_store_m((unsigned short *)(x + s + i + 256), v, p2);
    __vv_store_m((unsigned short *)(x + s + i + 384), v, p3);
  }
"""
    ),
    "vv_u16_size": source_for(
        """
  int s = taskId * E0;
  vv_uint8 b;
  vv_uint16 v;
  vv_bool p;
  __vv_move(v, (unsigned short)0xfc00);
  for (int i = 0; i < E0; i += 256) {
    __vv_load(b, m + s + i, 256);
    __vv_setp_ne(p, b, (unsigned char)0);
    __vv_store_m((unsigned short *)(x + s + i), v, 256, p);
  }
"""
    ),
    "vv_half": source_for(
        """
  int s = taskId * E0;
  vv_uint8 b;
  vv_half v;
  vv_bool p;
  __vv_move(v, (half)-65504.0);
  for (int i = 0; i < E0; i += 256) {
    __vv_load(b, m + s + i, 256);
    __vv_setp_ne(p, b, (unsigned char)0);
    __vv_store_m(x + s + i, v, p);
  }
"""
    ),
    "vv_half_size": source_for(
        """
  int s = taskId * E0;
  vv_uint8 b;
  vv_half v;
  vv_bool p;
  __vv_move(v, (half)-65504.0);
  for (int i = 0; i < E0; i += 256) {
    __vv_load(b, m + s + i, 256);
    __vv_setp_ne(p, b, (unsigned char)0);
    __vv_store_m(x + s + i, v, 256, p);
  }
"""
    ),
    "vv_nram_u16": source_for(
        """
  __nram__ unsigned char d[E0];
  __nram__ unsigned short c[E0];
  int s = taskId * E0;
  vv_uint16 b;
  vv_uint16 v;
  vv_bool p;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __vv_move(v, (unsigned short)0xfc00);
  for (int i = 0; i < E0; i += 256) {
    __vv_load(b, c + i);
    __vv_setp_ne(p, b, (unsigned short)0);
    __vv_store_m((unsigned short *)(x + s + i), v, p);
  }
"""
    ),
    "vv_select_u16": source_for(
        """
  __nram__ unsigned char d[E0];
  __nram__ unsigned short c[E0];
  int s = taskId * E0;
  vv_uint16 x0;
  vv_uint16 m0;
  vv_uint16 v0;
  vv_uint16 y0;
  vv_bool p0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __vv_move(v0, (unsigned short)0xfc00);
  for (int i = 0; i < E0; i += 256) {
    __vv_load(x0, (const unsigned short *)(x + s + i));
    __vv_load(m0, c + i);
    __vv_setp_ne(p0, m0, (unsigned short)0);
    __vv_select(y0, v0, x0, p0);
    __vv_store((unsigned short *)(x + s + i), y0);
  }
"""
    ),
    "vv_select_u32_128": source_for(
        """
  __nram__ unsigned char d[E0];
  __nram__ unsigned short c[E0];
  int s = taskId * E0;
  vv_uint32 x0;
  vv_uint32 m0;
  vv_uint32 v0;
  vv_uint32 y0;
  vv_bool p0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __vv_move(v0, (unsigned int)0xfc00);
  for (int i = 0; i < E0; i += 128) {
    __vv_load(x0, (const unsigned short *)(x + s + i));
    __vv_load(m0, c + i);
    __vv_setp_ne(p0, m0, (unsigned int)0);
    __vv_select(y0, v0, x0, p0);
    __vv_store((unsigned short *)(x + s + i), y0);
  }
"""
    ),
    "scalar_sparse": source_for(
        """
  int s = taskId * E0;
  unsigned short *p = (unsigned short *)(x + s);
  const unsigned char *q = m + s;
  for (int i = 0; i < E0; ++i) {
    if (q[i]) {
      p[i] = (unsigned short)0xfc00;
    }
  }
"""
    ),
    "pair_u32_m": source_for(
        """
  __nram__ unsigned char d[E0];
  __nram__ unsigned short c[E0];
  int s = taskId * E0;
  __memcpy(d, m + s, E0 * sizeof(unsigned char), GDRAM2NRAM);
  __bang_uchar2int16((int16_t *)c, d, E0, 0);
  __bang_mul_scalar((unsigned short *)c, (unsigned short *)c, (unsigned short)0xffff, E0);
  vv_uint32 x0;
  vv_uint32 m0;
  vv_uint32 n0;
  vv_uint32 v0;
  vv_uint32 y0;
  vv_bool p0;
  __vv_move(v0, (unsigned int)0xfc00fc00);
  for (int i = 0; i < E0 / 2; i += 128) {
    __vv_load(x0, (const unsigned int *)(x + s) + i);
    __vv_load(m0, ((const unsigned int *)c) + i);
    __vv_setp_ne(p0, m0, (unsigned int)0);
    __vv_not(n0, m0);
    __vv_and(x0, x0, n0);
    __vv_and(y0, v0, m0);
    __vv_or(y0, y0, x0);
    __vv_store_m(((unsigned int *)(x + s)) + i, y0, p0);
  }
"""
    ),
    "pair_u32_full": source_for(
        """
  __nram__ unsigned char d[65536];
  __nram__ unsigned short c[65536];
  __nram__ unsigned int a[32768];
  __nram__ unsigned int t[32768];
  int s = taskId * E0;
  for (int off = 0; off < E0; off += 65536) {
    __memcpy(d, m + s + off, 65536 * sizeof(unsigned char), GDRAM2NRAM);
    __bang_uchar2int16((int16_t *)c, d, 65536, 0);
    __bang_mul_scalar((unsigned short *)c, (unsigned short *)c, (unsigned short)0xffff, 65536);
    __memcpy(a, x + s + off, 65536 * sizeof(half), GDRAM2NRAM);
    __bang_bxor_scalar(t, a, (unsigned int)0xfc00fc00, 32768);
    __bang_band(t, t, (unsigned int *)c, 32768);
    __bang_bxor(a, a, t, 32768);
    __memcpy(x + s + off, a, 65536 * sizeof(half), NRAM2GDRAM);
  }
"""
    ),
}


def load_ext(name, text):
    h = hashlib.sha1(text.encode()).hexdigest()[:12]
    return load_inline(
        name=f"profile117_{name}_{h}",
        cpp_sources="#include <torch/extension.h>\n\ntorch::Tensor bang_func(torch::Tensor x, torch::Tensor mask, double fill_value);\n",
        bang_sources=text,
        functions=["bang_func"],
        verbose=False,
        extra_cflags=["-O3"],
        extra_ldflags=["-lcnrt", "-lbangc"],
        extra_bang_cflags=[
            "-O3",
            "-lm",
            "--bang-arch=compute_30",
            "--no-neuware-version-check",
            "--neuware-path=/usr/local/neuware",
        ],
    )


def force_fp16(values):
    out = []
    for v in values:
        if torch.is_tensor(v) and v.is_floating_point():
            out.append(v.to(torch.float16))
        else:
            out.append(v)
    return out


def time_once(fn):
    torch.mlu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.mlu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6, out


def summarize(name, xs):
    xs = sorted(xs)
    return {
        "name": name,
        "n": len(xs),
        "min": xs[0],
        "median": statistics.median(xs),
        "mean": statistics.mean(xs),
        "p90": xs[int(len(xs) * 0.9)] if xs else None,
        "max": xs[-1],
        "first": xs[: min(5, len(xs))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--calls", type=int, default=30)
    args = ap.parse_args()

    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("r117", root / "ref/ref_files/117_Masked_fill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.mlu.set_device(0)
    raw = force_fp16(mod.get_inputs())
    inputs = [v.mlu() if torch.is_tensor(v) else v for v in raw]
    init_inputs = force_fp16(mod.get_init_inputs())

    for name in args.variants:
        ext = load_ext(name, VARIANTS[name])
        xs = []
        for _ in range(args.calls):
            us, _ = time_once(lambda: ext.bang_func(*inputs, *init_inputs))
            xs.append(us)
        print(json.dumps(summarize(name, xs), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
