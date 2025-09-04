import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A):
    # A: (n,n) dense 0/1 or weighted; add I and symmetric normalize
    n = A.shape[0]
    A_hat = A + torch.eye(n, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(-1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv = torch.diag(deg_inv_sqrt)
    return D_inv @ A_hat @ D_inv

class GCNEncoder(nn.Module):
    """Lightweight GNN encoder (GCN-style) to avoid external deps.
    If you want GAT, you can replace this with a GAT implementation.
    """
    def __init__(self, d_in, h, d_out, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_in, h)
        self.lin2 = nn.Linear(h, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        A_norm = normalize_adj(A)
        H = A_norm @ F.relu(self.lin1(X))
        H = self.dropout(H)
        Z = A_norm @ self.lin2(H)
        return Z
