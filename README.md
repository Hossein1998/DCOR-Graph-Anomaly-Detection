# DCOR: Dual Contrastive Reconstruction for Graph Anomaly Detection

Official implementation of **DCOR** — dual autoencoders with reconstruction-level contrast (RLC) for anomaly detection in attributed graphs.

> This repository generalizes our single-dataset prototype to support **multiple datasets** (Amazon, Facebook, Flickr, ACM, Reddit, Enron) via **YAML configs**. It includes a clean package layout, augmentation operators, learnable margin, training/eval scripts, and ready-to-run configs.

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets
Place raw files under `data/raw/<dataset>/` (e.g., `.mat` or `.npz` containing `A`, `X`, and optionally `y`). Use the helper to convert `.mat` to `.npz`:
```bash
python scripts/process_mat_to_npz.py --in data/raw/amazon/amazon.mat --out data/processed/amazon.npz
```

## Quick Start (Amazon)
```bash
python train.py --dataset amazon --config configs/amazon.yaml
python eval.py  --dataset amazon --ckpt outputs/amazon/best.ckpt
```

## Key ideas
- Shared GNN encoder; structure decoder (inner product) + attribute decoder (linear).
- Reconstruction-level contrast with **learnable margin `m`**.
- Configurable **augmentation budgets** for structure/features.
- AUROC-based evaluation with paper-style anomaly score.

## Citation
If you use this code or processed datasets, please cite the DCOR paper.

## License
MIT
