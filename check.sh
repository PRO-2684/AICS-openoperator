#!/bin/bash

set -euo pipefail

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$NEUWARE_HOME/lib64"
export PATH="$PATH:$NEUWARE_HOME/bin"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

usage() {
  cat <<'EOF'
Usage:
  ./check.sh <code.mlu>

Behavior:
  compile the .mlu file with flags close to the remote evaluator
  then build/load a task-specific extension wrapper and compare against the task reference
EOF
}

if [ $# -ne 1 ]; then
  usage
  exit 1
fi

MLU_SOURCE="$1"

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

echo "[check] building torch extension and checking against task reference"
source /torch/venv3/pytorch/bin/activate
WRAPPER_SO="$BUILD_DIR/${TARGET}_smoke.so"
python scripts/run_reference_check.py \
  --source "$MLU_SOURCE" \
  --object "$OBJ_PATH" \
  --wrapper-so "$WRAPPER_SO"
