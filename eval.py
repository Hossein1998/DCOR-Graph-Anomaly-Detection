import argparse, yaml, os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from dcor.data import load_dataset
from dcor.models.encoder import GCNEncoder
from dcor.models.decoders import StructDecoder, AttrDecoder

def anomaly_score(A, Ahat, X, Xhat, alpha=0.5):
    a = torch.sum((X - Xhat)**2, dim=-1)
    b = torch.sum((A - Ahat)**2, dim=-1)
    return alpha * a + (1.0 - alpha) * b

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    args = ap.parse_args()

    A, X, y = load_dataset(args.dataset)
    n, d = X.shape
    state = torch.load(args.ckpt, map_location='cpu')
    cfg = state.get('cfg', {})
    enc = GCNEncoder(d_in=d, h=cfg.get('h',256), d_out=cfg.get('dz',128))
    decA = StructDecoder()
    decX = AttrDecoder(cfg.get('dz',128), d)
    enc.load_state_dict(state['enc'])
    decA.load_state_dict(state['decA'])
    decX.load_state_dict(state['decX'])

    with torch.no_grad():
        Z = enc(X, A)
        Ahat = decA(Z)
        Xhat = decX(Z)
        s = anomaly_score(A, Ahat, X, Xhat, alpha=args.alpha).numpy()
        if y is not None:
            auc = roc_auc_score(y.numpy(), s)
            print(f"AUROC: {auc:.4f}")
        else:
            print("Scores computed (no labels found).")
