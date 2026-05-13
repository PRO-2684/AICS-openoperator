#!/bin/bash

set -euo pipefail

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$NEUWARE_HOME/lib64"
export PATH="$PATH:$NEUWARE_HOME/bin"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

usage() {
  cat <<'EOF'
Usage:
  ./check.sh [options] <code.mlu>

Options:
  -v, --verbose       print full build/check logs
  --keep             keep the tmpfs build directory after exit
  --device N         use MLU device N, default 0
  --max-jobs N       set MAX_JOBS for extension build
  -h, --help         show this help

Default:
  success: print one compact PASS line only
  failure: print one compact FAIL line plus the last log lines to stderr
EOF
}

VERBOSE=0
KEEP=0
DEVICE=0
MAX_JOBS_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --verbose|-v)
      VERBOSE=1
      shift
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    --device)
      DEVICE="${2:?missing device id}"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS_ARG="${2:?missing MAX_JOBS value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      echo "FAIL unknown_option option=$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -ne 1 ]; then
  usage >&2
  exit 2
fi

MLU_SOURCE="$1"

if [ ! -f "$MLU_SOURCE" ]; then
  echo "FAIL file_not_found source=$MLU_SOURCE"
  exit 1
fi

if [ ! -d /dev/shm ]; then
  echo "FAIL missing_tmpfs path=/dev/shm"
  exit 1
fi

MLU_SOURCE_ABS="$(python - "$MLU_SOURCE" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"

TARGET="$(basename "${MLU_SOURCE%.mlu}")"

CACHE_KEY="$(python - "$MLU_SOURCE_ABS" <<'PY'
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1])
st = p.stat()
s = f"{p}:{st.st_mtime_ns}:{st.st_size}".encode()
print(hashlib.sha1(s).hexdigest()[:16])
PY
)"

SHM_ROOT="${CHECK_SHM_ROOT:-/dev/shm/aics_openoperator_check}"
BUILD_DIR="$SHM_ROOT/${TARGET}_${CACHE_KEY}"

mkdir -p "$BUILD_DIR"

export TMPDIR="$BUILD_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCH_EXTENSIONS_DIR="$BUILD_DIR/torch_extensions"
export XDG_CACHE_HOME="$BUILD_DIR/xdg_cache"

if [ -n "$MAX_JOBS_ARG" ]; then
  export MAX_JOBS="$MAX_JOBS_ARG"
else
  export MAX_JOBS="${MAX_JOBS:-1}" # If it is not 1, a compilation error may occur.
fi

mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

LOG="$BUILD_DIR/check.log"

cleanup() {
  if [ "$KEEP" != "1" ]; then
    rm -rf "$BUILD_DIR"
  else
    echo "KEEP build_dir=$BUILD_DIR log=$LOG"
  fi
}
trap cleanup EXIT

source /torch/venv3/pytorch/bin/activate

PY_ARGS=(
  scripts/run_reference_check.py
  --source "$MLU_SOURCE_ABS"
  --device "$DEVICE"
  --quiet
)

if [ -n "$MAX_JOBS_ARG" ]; then
  PY_ARGS+=(--max-jobs "$MAX_JOBS_ARG")
fi

if [ "$VERBOSE" = "1" ]; then
  PY_ARGS+=(--verbose)
  python "${PY_ARGS[@]}"
else
  if python "${PY_ARGS[@]}" >"$LOG" 2>&1; then
    tail -n 1 "$LOG"
  else
    code=$?
    echo "FAIL source=$(basename "$MLU_SOURCE")"
    tail -n 120 "$LOG" >&2 || true
    exit "$code"
  fi
fi
