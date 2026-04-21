#!/bin/bash
# https://gitservice.cstcloud.cn/kcxain/BangcTutorial/src/branch/master/01_vecadd/build_eval.sh

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <code.mlu>"
  exit 1
fi

MLU_SOURCE=$1
if [[ "$MLU_SOURCE" == *.mlu ]]; then
  TARGET="${MLU_SOURCE%.mlu}"
else
  TARGET="$MLU_SOURCE"
fi

cncc "${MLU_SOURCE}" \
    -I /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include \
    -I /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
    # -I /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/TH \
    # -I /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/THC \
    -o "target/${TARGET}" --bang-mlu-arch=mtp_592 -O3 -lm
# mtp_372

target/${TARGET}
