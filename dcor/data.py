import os, math, random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import torch

# -------------------- IO --------------------
def load_graph(path):
    """Load .npz or .mat with keys: A, X, (optional) y.
    Returns dense torch tensors A (0/1), X (float), y (long or None).
    """
    if path.endswith('.npz'):
        z = np.load(path, allow_pickle=True)
        A = z['A']; X = z['X']; y = z.get('y', None)
    elif path.endswith('.mat'):
        z = loadmat(path)
        A = z['A']; X = z['X']
        y = z.get('y', None)
    else:
        raise ValueError('Unsupported file: ' + path)

    if sp.issparse(A): A = A.todense()
    A = np.asarray(A).astype(np.float32)
    X = np.asarray(X).astype(np.float32)
    y = None if y is None else np.asarray(y).squeeze()

    # ensure symmetric, zero diag
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A), torch.from_numpy(X), (None if y is None else torch.from_numpy(y))

# -------------------- Augmentations --------------------
def node_isolation(A, p):
    """Drop all incident edges for a fraction p of nodes."""
    n = A.shape[0]
    k = int(max(0, round(p * n)))
    idx = torch.randperm(n)[:k]
    A2 = A.clone()
    A2[idx, :] = 0.0
    A2[:, idx] = 0.0
    sA = torch.zeros(n, dtype=torch.long, device=A.device)
    sA[idx] = 1
    return A2, sA

def random_shortcuts(A, p, num_per_node=1):
    """Add random edges for a fraction p of nodes (skip self/dupes)."""
    n = A.shape[0]
    k = int(max(0, round(p * n)))
    idx = torch.randperm(n)[:k]
    A2 = A.clone()
    for i in idx:
        for _ in range(num_per_node):
            j = torch.randint(0, n, (1,), device=A.device).item()
            if j == i: 
                continue
            A2[i, j] = 1.0
            A2[j, i] = 1.0
    s = torch.zeros(n, dtype=torch.long, device=A.device)
    s[idx] = 1
    return A2, s

def feature_scaling(X, p, alpha=0.5):
    """Scale features for a fraction p of nodes by factor alpha or 1/alpha."""
    n = X.shape[0]
    k = int(max(0, round(p * n)))
    idx = torch.randperm(n)[:k]
    X2 = X.clone()
    factor = alpha if random.random() < 0.5 else (1.0/alpha)
    X2[idx] = X2[idx] * factor
    s = torch.zeros(n, dtype=torch.long, device=X.device)
    s[idx] = 1
    return X2, s

def feature_copying(X, p, pool=64):
    """Copy features from random pool nodes to a fraction p of nodes."""
    n = X.shape[0]
    k = int(max(0, round(p * n)))
    if k == 0:
        return X.clone(), torch.zeros(n, dtype=torch.long, device=X.device)
    idx = torch.randperm(n)[:k]
    pool_idx = torch.randperm(n)[:min(pool, n)]
    X2 = X.clone()
    src = pool_idx[torch.randint(0, pool_idx.shape[0], (k,), device=X.device)]
    X2[idx] = X[src]
    s = torch.zeros(n, dtype=torch.long, device=X.device)
    s[idx] = 1
    return X2, s

def make_augmented_view(A, X, pA=0.15, pX=0.20):
    """Compose structural + feature augmentations; returns A', X', sA, sX."""
    A1, sA1 = node_isolation(A, pA * 0.5)
    A2, sA2 = random_shortcuts(A1, pA * 0.5, num_per_node=1)
    sA = torch.clamp(sA1 + sA2, max=1)

    X1, sX1 = feature_scaling(X, pX * 0.5, alpha=0.7)
    X2, sX2 = feature_copying(X1, pX * 0.5, pool=64)
    sX = torch.clamp(sX1 + sX2, max=1)
    return A2, X2, sA, sX

# -------------------- Dataset registry --------------------
DEFAULT_DATA = {
    'amazon':    'data/processed/amazon.npz',
    'facebook':  'data/processed/facebook.npz',
    'flickr':    'data/processed/flickr.npz',
    'acm':       'data/processed/acm.npz',
    'reddit':    'data/processed/reddit.npz',
    'enron':     'data/processed/enron.npz',
}

def load_dataset(name, path=None):
    if path is None:
        path = DEFAULT_DATA.get(name.lower())
        if path is None:
            raise ValueError(f'Unknown dataset {name} and no path provided.')
    return load_graph(path)
