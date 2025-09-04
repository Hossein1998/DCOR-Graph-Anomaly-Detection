import argparse, os, yaml, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dcor.data import load_dataset, make_augmented_view
from dcor.models.encoder import GCNEncoder
from dcor.models.decoders import StructDecoder, AttrDecoder
from dcor.losses import RLCLoss, frob

def anomaly_score(A, Ahat, X, Xhat, alpha=0.5):
    # per-node score: alpha * ||Xi-Xhat_i||^2 + (1-alpha) * ||Ai-Ahat_i||^2
    a = torch.sum((X - Xhat)**2, dim=-1)
    b = torch.sum((A - Ahat)**2, dim=-1)
    return alpha * a + (1.0 - alpha) * b

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = cfg['dataset']
    A, X, y = load_dataset(ds)
    A, X = A.to(device), X.to(device)
    n, d = X.shape
    dz = cfg['dz']
    h  = cfg['h']

    enc = GCNEncoder(d_in=d, h=h, d_out=dz, dropout=cfg.get('dropout', 0.1)).to(device)
    decA = StructDecoder().to(device)
    decX = AttrDecoder(dz, d).to(device)
    rlc = RLCLoss(margin_init=cfg.get('margin_init', 0.2)).to(device)

    params = list(enc.parameters()) + list(decA.parameters()) + list(decX.parameters()) + list(rlc.parameters())
    opt = torch.optim.Adam(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    outdir = cfg.get('outdir', f'outputs/{ds}')
    os.makedirs(outdir, exist_ok=True)

    best = -1.0
    for epoch in range(1, cfg['epochs'] + 1):
        enc.train(); decA.train(); decX.train(); rlc.train()
        opt.zero_grad()

        # Build two views
        A2, X2, sA, sX = make_augmented_view(A, X, pA=cfg['pA'], pX=cfg['pX'])
        sA, sX = sA.to(device), sX.to(device)

        # Encode both views
        Z1 = enc(X,  A)
        Z2 = enc(X2, A2)

        # Decode
        A1_hat = decA(Z1)
        X1_hat = decX(Z1)
        A2_hat = decA(Z2)
        X2_hat = decX(Z2)

        # Reconstruction loss (structure + attributes)
        lam = cfg.get('lambda_modal', 0.5)
        L_rec = lam * frob(A - A1_hat) + (1.0 - lam) * frob(X - X1_hat)

        # RLC (cross-view)
        L_rlc = rlc(A1_hat, A2_hat, X1_hat, X2_hat, sA, sX)

        L_total = cfg['lambda_rec'] * L_rec + cfg['lambda_rlc'] * L_rlc
        L_total.backward()
        opt.step()

        if epoch % cfg.get('log_every', 20) == 0 or epoch == 1:
            enc.eval(); decA.eval(); decX.eval()
            with torch.no_grad():
                Z = enc(X, A)
                Ahat = decA(Z)
                Xhat = decX(Z)
                score = anomaly_score(A, Ahat, X, Xhat, alpha=cfg.get('alpha', 0.5))
                if y is not None:
                    y_np = y.detach().cpu().numpy()
                    s_np = score.detach().cpu().numpy()
                    auc = roc_auc_score(y_np, s_np)
                else:
                    auc = float('nan')
                print(f'Epoch {epoch:04d} | L_total {L_total.item():.4f} | L_rec {L_rec.item():.4f} | L_rlc {L_rlc.item():.4f} | AUROC {auc:.4f}')
                if not np.isnan(auc) and auc > best:
                    best = auc
                    torch.save({'enc': enc.state_dict(),
                                'decA': decA.state_dict(),
                                'decX': decX.state_dict(),
                                'rlc': rlc.state_dict(),
                                'cfg': cfg},
                               os.path.join(outdir, 'best.ckpt'))
    return outdir, best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='amazon|facebook|flickr|acm|reddit|enron')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['dataset'] = args.dataset
    train(cfg)
