import numpy as np
import torch
from torch.utils.data import Dataset


class VectorDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray = None):
        if len(features) != len(targets):
            raise ValueError("Number of records in 'features' should be the same as in 'targets'")

        self.X = features
        self.y = targets

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.X[idx])
        if self.y is None:
            return feat
        target = torch.from_numpy(self.y[idx])
        return feat, target
