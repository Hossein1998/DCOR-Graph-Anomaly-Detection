import torch
import torch.nn as nn

class StructDecoder(nn.Module):
    def forward(self, Z):
        # inner-product decoder -> edge likelihoods in [0,1]
        return torch.sigmoid(Z @ Z.T)

class AttrDecoder(nn.Module):
    def __init__(self, dz, d):
        super().__init__()
        self.lin = nn.Linear(dz, d)

    def forward(self, Z):
        return self.lin(Z)
