"""
bangc_torch_tester.py — BangC vs PyTorch 精度与性能对比测试

自动行为：
  脚本启动时在自身所在目录下查找 ref_times.json。
  如果存在，根据 -t 传入的 torch 模型文件名匹配问题名，
  找到预存时间则跳过 PyTorch 性能测试，直接用预存时间计算加速比。
  找不到则退回现场测量，完全向后兼容。
"""

import sys
import json
import traceback
import importlib.util
from pathlib import Path
import time

import torch
import torch_mlu
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# ── 自动加载预存参考时间 ──────────────────────────────
REF_TIMES: dict[str, dict] = {}
_ref_path = SCRIPT_DIR / "ref_times.json"
if _ref_path.exists():
    try:
        with open(_ref_path, encoding="utf-8") as _f:
            REF_TIMES = json.load(_f)
    except Exception:
        REF_TIMES = {}


def _lookup_ref_time(torch_model_path: str) -> float | None:
    """
    根据 torch 模型文件名在 REF_TIMES 中查找预存时间。
    匹配规则：文件名去掉 .py 后缀作为 key。
    例如 /path/to/ref/001_LeakyReLU.py → key "001_LeakyReLU"
    """
    if not REF_TIMES:
        return None
    stem = Path(torch_model_path).stem  # "001_LeakyReLU"
    info = REF_TIMES.get(stem)
    if info and "torch_us" in info:
        return info["torch_us"]
    return None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _force_fp16_tensors(values):
    """
    仅将输入中的浮点 Tensor 统一转为 float16。
    整型/布尔 Tensor 与 Python 标量保持不变。
    """
    normalized = []
    for value in values:
        if torch.is_tensor(value) and value.is_floating_point():
            normalized.append(value.to(torch.float16))
        else:
            normalized.append(value)
    return normalized


def _promote_nested_tensors_to_float32(value):
    if torch.is_tensor(value):
        return value.to(torch.float32) if value.is_floating_point() else value
    if isinstance(value, list):
        return [_promote_nested_tensors_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_promote_nested_tensors_to_float32(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _promote_nested_tensors_to_float32(item) for key, item in value.items()
        }
    return value


def _move_nested_tensors_to_mlu(value):
    if torch.is_tensor(value):
        return value.mlu()
    if isinstance(value, list):
        return [_move_nested_tensors_to_mlu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors_to_mlu(item) for item in value)
    if isinstance(value, dict):
        return {key: _move_nested_tensors_to_mlu(item) for key, item in value.items()}
    return value


def _move_nested_tensors_to_mlu_fp32(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(torch.float32).mlu()
        return value.mlu()
    if isinstance(value, list):
        return [_move_nested_tensors_to_mlu_fp32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors_to_mlu_fp32(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors_to_mlu_fp32(item) for key, item in value.items()
        }
    return value


def _move_unregistered_tensor_attrs_to_mlu(module):
    """
    `.mlu()` 只会自动迁移 Parameter 和 buffer。
    这里额外兜底迁移普通 Tensor 属性，避免 kernel/bias 等留在 CPU。
    """
    for submodule in module.modules():
        registered_names = (
            set(submodule._parameters)
            | set(submodule._buffers)
            | set(submodule._modules)
        )
        for name, value in vars(submodule).items():
            if name in registered_names:
                continue
            moved_value = _move_nested_tensors_to_mlu(value)
            if moved_value is not value:
                setattr(submodule, name, moved_value)

    return module


def _promote_module_to_float32_mlu(module):
    module = module.to(torch.float32).mlu()
    for submodule in module.modules():
        registered_names = (
            set(submodule._parameters)
            | set(submodule._buffers)
            | set(submodule._modules)
        )
        for name, value in vars(submodule).items():
            if name in registered_names:
                continue
            moved_value = _move_nested_tensors_to_mlu_fp32(value)
            if moved_value is not value:
                setattr(submodule, name, moved_value)
    return module


def _tensor_max_abs_diff(
    expected: torch.Tensor, actual: torch.Tensor, path: str
) -> float:
    if expected.shape != actual.shape:
        raise ValueError(
            f"{path} shape mismatch: {tuple(expected.shape)} vs {tuple(actual.shape)}"
        )

    if expected.numel() == 0:
        return 0.0

    expected_cpu = expected.detach().cpu()
    actual_cpu = actual.detach().cpu()

    if expected_cpu.dtype == torch.bool or actual_cpu.dtype == torch.bool:
        return float(
            torch.logical_xor(expected_cpu.bool(), actual_cpu.bool()).any().item()
        )

    if expected_cpu.is_floating_point() or actual_cpu.is_floating_point():
        return float((expected_cpu.float() - actual_cpu.float()).abs().max().item())

    return float(
        (expected_cpu.to(torch.int64) - actual_cpu.to(torch.int64)).abs().max().item()
    )


def _max_abs_diff(expected, actual, path="output") -> float:
    if torch.is_tensor(expected) and torch.is_tensor(actual):
        return _tensor_max_abs_diff(expected, actual, path)

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            raise ValueError(
                f"{path} length mismatch: {len(expected)} vs {len(actual)}"
            )
        if not expected:
            return 0.0
        return max(
            _max_abs_diff(exp_item, act_item, f"{path}[{idx}]")
            for idx, (exp_item, act_item) in enumerate(zip(expected, actual))
        )

    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(
                f"{path} key mismatch: {sorted(expected.keys())} vs {sorted(actual.keys())}"
            )
        if not expected:
            return 0.0
        return max(
            _max_abs_diff(expected[key], actual[key], f"{path}[{key!r}]")
            for key in expected
        )

    if isinstance(expected, (bool, int, float)) and isinstance(
        actual, (bool, int, float)
    ):
        return float(abs(float(expected) - float(actual)))

    raise TypeError(
        f"{path} unsupported output types: {type(expected).__name__} vs {type(actual).__name__}"
    )


def _svd_reconstruct(output):
    u, s, vh = output
    return (u * s.unsqueeze(-2)) @ vh


def _problem_specific_max_abs_diff(
    torch_model_path: str, inputs, expected, actual
) -> float:
    stem = Path(torch_model_path).stem

    if stem == "075_TopK":
        # TopK values define the numeric result; indices may differ on ties across backends.
        return _max_abs_diff(expected[0], actual[0], "output.values")

    if stem == "101_Max_Pool_2D_with_indices":
        # Pooled values are the stable numeric target; indices may differ on tied maxima.
        return _max_abs_diff(expected[0], actual[0], "output.values")

    if stem == "092_SVD_decomposition":
        # U/Vh are not unique up to sign flips; compare singular values and reconstructed matrix instead.
        s_diff = _max_abs_diff(expected[1], actual[1], "output.S")
        recon_diff = _max_abs_diff(
            _svd_reconstruct(expected),
            _svd_reconstruct(actual),
            "output.reconstruction",
        )
        return max(s_diff, recon_diff)

    return _max_abs_diff(expected, actual)


def _get_problem_threshold(torch_model_path: str) -> float:
    stem = Path(torch_model_path).stem

    if stem == "096_LocalResponseNorm":
        return 1e-3

    return 1e-2


def benchmark_hardware_time_once(fn, inputs):
    """
    单次性能测量，返回本次调用的硬件时间（us）与输出结果。
    """
    torch.mlu.synchronize()
    t0 = time.perf_counter()
    output = fn(*inputs)
    torch.mlu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6, output


def _build_input_pairs(torch_mod):
    raw_inputs = _force_fp16_tensors(torch_mod.get_inputs())
    torch_inputs = [
        _move_nested_tensors_to_mlu_fp32(value) if torch.is_tensor(value) else value
        for value in raw_inputs
    ]
    bangc_inputs = [
        value.mlu() if torch.is_tensor(value) else value for value in raw_inputs
    ]
    return raw_inputs, torch_inputs, bangc_inputs


def _run_correctness_once(
    torch_model_path: str, model, model_new, raw_inputs, torch_inputs, bangc_inputs
):
    try:
        with torch.no_grad():
            torch_out = model(*torch_inputs)
            torch.mlu.synchronize()
    except Exception as e:
        print(f"[ERROR] PyTorch 推理失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        with torch.no_grad():
            bangc_out = model_new(*bangc_inputs)
            torch.mlu.synchronize()
    except Exception as e:
        print(f"[RUNTIME ERROR] BangC 推理失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(3)

    try:
        return _problem_specific_max_abs_diff(
            torch_model_path, raw_inputs, torch_out, bangc_out
        )
    except Exception as e:
        print(f"[ERROR] 输出对比失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def run(torch_model_path, bangc_model_path):
    torch.set_default_dtype(torch.float16)
    # ── 加载 PyTorch 参考模型 ──
    try:
        torch_mod = load_module("torch_model", torch_model_path)
    except Exception as e:
        print(f"[ERROR] 加载 PyTorch 参考模型失败: {torch_model_path}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ── 加载 BangC 模型 ──
    try:
        bangc_mod = load_module("bangc_model", bangc_model_path)
    except Exception as e:
        print(
            f"[COMPILE ERROR] BangC 模型加载/编译失败: {bangc_model_path}",
            file=sys.stderr,
        )
        print(f"错误类型: {type(e).__name__}", file=sys.stderr)
        print(f"错误信息: {e}", file=sys.stderr)
        print("", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)

    # ── 实例化模型 ──
    try:
        init_inputs = _force_fp16_tensors(torch_mod.get_init_inputs())
        ref_init_inputs = _promote_nested_tensors_to_float32(init_inputs)
        model = _promote_module_to_float32_mlu(torch_mod.Model(*ref_init_inputs))
        model_perf = _move_unregistered_tensor_attrs_to_mlu(
            torch_mod.Model(*init_inputs).mlu()
        )
    except Exception as e:
        print(f"[ERROR] PyTorch Model 实例化失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        model_new = _move_unregistered_tensor_attrs_to_mlu(
            bangc_mod.ModelNew(*init_inputs).mlu()
        )
    except Exception as e:
        print(f"[COMPILE ERROR] BangC ModelNew 实例化失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)

    # ── 单次性能测量与正确性评估 ──
    raw_inputs, torch_inputs, bangc_inputs = _build_input_pairs(torch_mod)

    saved_ref = _lookup_ref_time(torch_model_path)
    if saved_ref is not None:
        torch_time = saved_ref
        print(f"Ref time     : {torch_time:.3f} us (from ref_times.json)")
    else:
        try:
            with torch.no_grad():
                torch_time, _ = benchmark_hardware_time_once(model_perf, bangc_inputs)
        except Exception as e:
            print(f"[ERROR] PyTorch 性能测试失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    try:
        with torch.no_grad():
            bangc_time, bangc_out = benchmark_hardware_time_once(
                model_new, bangc_inputs
            )
    except Exception as e:
        print(f"[RUNTIME ERROR] BangC 推理失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(3)

    try:
        with torch.no_grad():
            torch_out = model(*torch_inputs)
            torch.mlu.synchronize()
        max_abs_diff = _problem_specific_max_abs_diff(
            torch_model_path, raw_inputs, torch_out, bangc_out
        )
    except Exception as e:
        print(f"[ERROR] 输出对比失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    passed = max_abs_diff < _get_problem_threshold(torch_model_path)

    if bangc_time == 0.0:
        print("bangc时间异常，请联系管理员")
        bangc_time = torch_time * 10000
        passed = False
    score = torch_time / bangc_time if passed else 0.0

    # ── 结构化 JSON 输出（供 worker 解析） ──
    result = {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "torch_us": torch_time,
        "bangc_us": bangc_time,
        "score": score,
    }
    print(f"@@RESULT@@{json.dumps(result)}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", required=True, help="PyTorch 模型文件")
    parser.add_argument("-b", required=True, help="BangC load_inline 模型文件")
    args = parser.parse_args()

    if not torch.mlu.is_available():
        print("MLU 不可用")
        sys.exit(1)

    run(args.t, args.b)
