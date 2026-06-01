#!/bin/bash
# 环境变量设置
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEUWARE_HOME/lib64
export PATH=$PATH:$NEUWARE_HOME/bin
export MLU_VISIBLE_DEVICES=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

set -euo pipefail

# 参数检查
if [ $# -ne 1 ]; then
  echo "Usage: $0 <mlucode 文件名，*.mlu>"
  exit 1
fi

MLU_SOURCE=$1
if [[ "$MLU_SOURCE" == *.mlu ]]; then
  TARGET="${MLU_SOURCE%.mlu}"
else
  TARGET="$MLU_SOURCE"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活环境
source /torch/venv3/pytorch/bin/activate

# 编译和运行
pushd "$SCRIPT_DIR" >/dev/null
echo "=== Compiling $MLU_SOURCE ==="
cncc "${MLU_SOURCE}" -o "${TARGET}" --bang-mlu-arch=mtp_592 -O3 -lm
if [ $? -eq 0 ]; then
    echo "=== Compilation successful ==="
    echo ""
    echo "=== Running ${TARGET} ==="
    ./"${TARGET}"
    echo ""
    echo "=== Execution completed ==="
else
    echo "=== Compilation failed ==="
    exit 1
fi
popd >/dev/null
