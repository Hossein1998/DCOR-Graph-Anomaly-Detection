import torch
import torch.nn as nn
import torch.nn.functional as F

def frob(X):
    return torch.linalg.matrix_norm(X, ord='fro')

def D_rows(U, V):
    # squared Euclidean per row
    return torch.sum((U - V)**2, dim=-1)

class RLCLoss(nn.Module):
    """Reconstruction-level contrast with learnable margin m."""
    def __init__(self, margin_init=0.2):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(float(margin_init)))

    def forward(self, A1, A2, X1, X2, sA, sX):
        # A1,A2: (n,n) reconstructions for two views
        # X1,X2: (n,d) reconstructions for two views
        # sA,sX: (n,) in {0,1} augmentation indicators
        La = torch.where(sA==0, D_rows(A1, A2), F.relu(self.m - D_rows(A1, A2)))
        Lx = torch.where(sX==0, D_rows(X1, X2), F.relu(self.m - D_rows(X1, X2)))
        return La.mean() + Lx.mean()
