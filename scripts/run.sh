#!/usr/bin/env bash
set -e
DATASET=${1:-amazon}
python train.py --dataset ${DATASET} --config configs/${DATASET}.yaml
python eval.py  --dataset ${DATASET} --ckpt outputs/${DATASET}/best.ckpt
