#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash tools/train_8npu.sh <config.yaml>

Examples:
  bash tools/train_8npu.sh config/OCID-VLG/pangu_crog_multiple_r50.yaml
  bash tools/train_8npu.sh config/OCID-VLG/pangu_drog.yaml
  bash tools/train_8npu.sh config/OCID-VLG/pangu_drogoff.yaml
  bash tools/train_8npu.sh config/OCID-VLG/pangu_etrg.yaml
  bash tools/train_8npu.sh config/grasp_tools/pangu_drogoff.yaml
  bash tools/train_8npu.sh config/vcot/pangu_drogoff.yaml
EOF
}

if [[ "$#" -ne 1 ]]; then
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="$1"
[[ -f "${CONFIG}" ]] || {
  echo "Config file not found: ${CONFIG}" >&2
  usage
  exit 2
}

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CROG_RUN_TIMESTAMP="${CROG_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S_%3N)}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if grep -Eq '^[[:space:]]*dataset[[:space:]]*:[[:space:]]*vcot([[:space:]]|$)' "${CONFIG}"; then
  DATASET_NAME="VCoT/Grasp-Anything"
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/graspanything-vcot}"
  SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/split/vcot}"
elif grep -Eqi '^[[:space:]]*dataset[[:space:]]*:[[:space:]]*grasp-?tools?([[:space:]]|$)' "${CONFIG}"; then
  DATASET_NAME="Grasp-Tools"
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/grasp-tools/aug_graspall_v2}"
  SPLIT_ROOT=""
else
  DATASET_NAME="OCID-VLG"
  DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
  SPLIT_ROOT=""
fi

[[ -d "${DATA_ROOT}" ]] || {
  echo "${DATASET_NAME} dataset directory not found: ${DATA_ROOT}" >&2
  exit 2
}

TRAIN_OPTS=(DATA.root_path "${DATA_ROOT}")
if [[ -n "${SPLIT_ROOT}" ]]; then
  for SPLIT_FILE in train.csv test_unseen.csv; do
    [[ -f "${SPLIT_ROOT}/${SPLIT_FILE}" ]] || {
      echo "VCoT split not found: ${SPLIT_ROOT}/${SPLIT_FILE}" >&2
      exit 2
    }
  done
  TRAIN_OPTS+=(DATA.split_root "${SPLIT_ROOT}")
fi

# DROG and DROG-OFF configs contain a DINO backbone; CROG configs do not.
# Use that model-owned field instead of relying on a filename convention.
if grep -Eq '^[[:space:]]*dino_pretrain[[:space:]]*:' "${CONFIG}"; then
  MODEL_FAMILY="PanguDROG/PanguDROG-OFF"
  CLIP_WEIGHT="${CLIP_WEIGHT:-${REPO_ROOT}/pretrain/ViT-B-16.pt}"
  DINO_WEIGHT="${DINO_WEIGHT:-${REPO_ROOT}/pretrain/dinov2_vitb14_reg4_pretrain.pth}"

  python3 tools/download_pretrained.py clip-vit-b16 --output "${CLIP_WEIGHT}"
  python3 tools/download_pretrained.py dinov2-vitb14-reg4 --output "${DINO_WEIGHT}"

  TRAIN_OPTS+=(
    TRAIN.clip_pretrain "${CLIP_WEIGHT}"
    TRAIN.dino_pretrain "${DINO_WEIGHT}"
  )
else
  MODEL_FAMILY="PanguCROG"
  CLIP_WEIGHT="${CLIP_WEIGHT:-${REPO_ROOT}/pretrain/RN50.pt}"

  python3 tools/download_pretrained.py clip-rn50 --output "${CLIP_WEIGHT}"

  TRAIN_OPTS+=(
    TRAIN.clip_pretrain "${CLIP_WEIGHT}"
  )
fi

if grep -Eq '^[[:space:]]*architecture[[:space:]]*:[[:space:]]*(pangu_)?etrg([[:space:]]|$)' "${CONFIG}"; then
  MODEL_FAMILY="PanguETRG"
  RESNET_WEIGHT="${RESNET_WEIGHT:-${REPO_ROOT}/pretrain/resnet18-f37072fd.pth}"
  python3 tools/download_pretrained.py resnet18 --output "${RESNET_WEIGHT}"
  TRAIN_OPTS+=(TRAIN.depth_pretrain "${RESNET_WEIGHT}")
fi

echo "[launch] config: ${CONFIG}"
echo "[launch] run timestamp: ${CROG_RUN_TIMESTAMP}"
echo "[launch] model family: ${MODEL_FAMILY}"
echo "[launch] dataset: ${DATASET_NAME}"
echo "[launch] data root: ${DATA_ROOT}"
[[ -z "${SPLIT_ROOT}" ]] || echo "[launch] split root: ${SPLIT_ROOT}"
echo "[launch] visible NPUs: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "[launch] torchrun processes on this node: ${NPROC_PER_NODE}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_crog.py \
  --config "${CONFIG}" \
  --opts \
  "${TRAIN_OPTS[@]}"
