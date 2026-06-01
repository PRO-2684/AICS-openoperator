#!/bin/bash
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEUWARE_HOME/lib64
export PATH=$PATH:$NEUWARE_HOME/bin
export MLU_VISIBLE_DEVICES=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

set -euo pipefail

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

source /torch/venv3/pytorch/bin/activate

pushd "$SCRIPT_DIR" >/dev/null
cncc "${MLU_SOURCE}" -o "${TARGET}" --bang-mlu-arch=mtp_592 -O3 -lm
./"${TARGET}"
popd >/dev/null
