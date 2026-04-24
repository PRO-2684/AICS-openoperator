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

cncc -c "${MLU_SOURCE}" -o "target/${TARGET}" -DTORCH_EXTENSION_NAME=LeakyReLU -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" \
    -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include \
    -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
    -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/TH \
    -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/THC \
    -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu/csrc \
    -isystem /torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu/csrc/api/include/torch_mlu \
    -isystem /usr/local/neuware/include -isystem /opt/py3.10/include/python3.10 \
    -D_GLIBCXX_USE_CXX11_ABI=0 --no-neuware-version-check \
    --bang-mlu-arch=mtp_372 \
    -fPIC --neuware-path=/usr/local/neuware -std=c++17

target/${TARGET}
