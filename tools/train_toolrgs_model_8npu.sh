#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-pangu_drog}"
CONFIG="config/OCID-VLG/${MODEL}.yaml"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EXP_NAME="${EXP_NAME:-${MODEL}_crog_protocol_8npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CROG_RUN_TIMESTAMP="${CROG_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S_%3N)}"

case "${MODEL}" in
  etrg|pangu_etrg)
    WEIGHTS=(clip-rn50 resnet18)
    ;;
  crogoff|pangu_crogoff|ggcnnclip|pangu_ggcnnclip|grconvnetclip|pangu_grconvnetclip|lgd|pangu_lgd|maplegrasp|pangu_maplegrasp)
    WEIGHTS=(clip-rn50)
    ;;
  drog|pangu_drog|drogoff|pangu_drogoff)
    WEIGHTS=(clip-vit-b16 dinov2-vitb14-reg4)
    ;;
  graspmamba|pangu_graspmamba)
    WEIGHTS=(clip-rn50 mambavision-t)
    ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    echo "Choose Pangu names: pangu_crogoff pangu_drog pangu_drogoff pangu_etrg pangu_ggcnnclip pangu_grconvnetclip pangu_graspmamba pangu_lgd pangu_maplegrasp" >&2
    exit 2
    ;;
esac

[[ -f "${CONFIG}" ]] || {
  echo "Model config not found: ${CONFIG}" >&2
  exit 2
}
[[ -d "${DATA_ROOT}" ]] || {
  echo "OCID-VLG dataset directory not found: ${DATA_ROOT}" >&2
  exit 2
}
[[ -f "${DATA_ROOT}/refer/multiple/train_expressions.json" ]] || {
  echo "OCID-VLG training expressions not found under: ${DATA_ROOT}" >&2
  exit 2
}

python3 tools/download_pretrained.py "${WEIGHTS[@]}"

echo "[launch] model: ${MODEL}"
echo "[launch] config: ${CONFIG}"
echo "[launch] run timestamp: ${CROG_RUN_TIMESTAMP}"
echo "[launch] protocol: CROG legacy"
echo "[launch] global batch size comes from YAML; processes: ${NPROC_PER_NODE}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_crog.py \
  --config "${CONFIG}" \
  --opts \
  DATA.root_path "${DATA_ROOT}" \
  TRAIN.exp_name "${EXP_NAME}"
