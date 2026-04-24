#!/bin/bash

set -euo pipefail

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$NEUWARE_HOME/lib64"
export PATH="$PATH:$NEUWARE_HOME/bin"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

usage() {
  cat <<'EOF'
Usage:
  ./check.sh <code.mlu> [--smoke]

Modes:
  default   compile the .mlu file with flags close to the remote evaluator
  --smoke   compile/load it as a torch extension and run a small float32/float16 correctness check on MLU
EOF
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage
  exit 1
fi

MLU_SOURCE="$1"
SMOKE_TEST=0
if [ $# -eq 2 ]; then
  if [ "$2" != "--smoke" ]; then
    usage
    exit 1
  fi
  SMOKE_TEST=1
fi

if [ ! -f "$MLU_SOURCE" ]; then
  echo "File not found: $MLU_SOURCE" >&2
  exit 1
fi

TARGET="$(basename "${MLU_SOURCE%.mlu}")"
BUILD_DIR="target"
OBJ_PATH="$BUILD_DIR/${TARGET}.mlu.o"
mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
  -DTORCH_EXTENSION_NAME="$TARGET"
  -DTORCH_API_INCLUDE_EXTENSION_H
  -DPYBIND11_COMPILER_TYPE=\"_gcc\"
  -DPYBIND11_STDLIB=\"_libstdcpp\"
  -DPYBIND11_BUILD_ABI=\"_cxxabi1011\"
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/TH
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/THC
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu/csrc
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu/csrc/api/include/torch_mlu
  -isystem /usr/local/neuware/include
  -isystem /opt/py3.10/include/python3.10
  -D_GLIBCXX_USE_CXX11_ABI=0
  --bang-arch=compute_30
  --no-neuware-version-check
  -fPIC
  --neuware-path=/usr/local/neuware
  -std=c++17
)

echo "[compile] $MLU_SOURCE -> $OBJ_PATH"
cncc -c "$MLU_SOURCE" -o "$OBJ_PATH" "${COMMON_FLAGS[@]}"
echo "[ok] compile succeeded"

if [ "$SMOKE_TEST" -eq 0 ]; then
  exit 0
fi

echo "[smoke] building torch extension and checking float32/float16 outputs"
source /torch/venv3/pytorch/bin/activate
WRAPPER_CPP="$BUILD_DIR/${TARGET}_smoke_binding.cpp"
WRAPPER_OBJ="$BUILD_DIR/${TARGET}_smoke_binding.o"
WRAPPER_SO="$BUILD_DIR/${TARGET}_smoke.so"

cat >"$WRAPPER_CPP" <<EOF
#include <torch/extension.h>

torch::Tensor bang_func(torch::Tensor input, double negative_slope);

PYBIND11_MODULE(${TARGET}_smoke, m) {
  m.def("bang_func", &bang_func, "LeakyReLU bang_func");
}
EOF

c++ -c "$WRAPPER_CPP" -o "$WRAPPER_OBJ" \
  -DTORCH_EXTENSION_NAME=${TARGET}_smoke \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -DPYBIND11_COMPILER_TYPE=\"_gcc\" \
  -DPYBIND11_STDLIB=\"_libstdcpp\" \
  -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" \
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include \
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/TH \
  -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/THC \
  -isystem /opt/py3.10/include/python3.10 \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -fPIC -std=c++17

c++ "$WRAPPER_OBJ" "$OBJ_PATH" -shared -o "$WRAPPER_SO" \
  -L/torch/venv3/pytorch/lib/python3.10/site-packages/torch/lib \
  -L/torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu/csrc/lib \
  -L/usr/local/neuware/lib64 \
  -lc10 -ltorch_cpu -ltorch -ltorch_python -ltorch_mlu -ltorch_mlu_python -lcnrt -lbangc

python - "$WRAPPER_SO" <<'PY'
import importlib.util
import pathlib
import sys

import torch
import torch_mlu  # noqa: F401

module_path = pathlib.Path(sys.argv[1]).resolve()

if not torch.mlu.is_available():
    raise SystemExit("MLU is not available, cannot run --smoke")

spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

device = "mlu"
slope = 0.1

for dtype in (torch.float32, torch.float16):
    x = torch.linspace(-3, 3, steps=257, device=device, dtype=dtype).reshape(1, -1)
    expected = torch.nn.functional.leaky_relu(x, negative_slope=slope)
    actual = module.bang_func(x, slope)
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    if not torch.allclose(actual.float().cpu(), expected.float().cpu(), atol=atol, rtol=rtol):
        diff = (actual.float() - expected.float()).abs().max().item()
        raise SystemExit(f"{dtype} smoke test failed, max abs diff={diff}")
    print(f"[ok] {dtype} smoke test passed")
PY
