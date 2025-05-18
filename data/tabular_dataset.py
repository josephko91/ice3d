import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols):
        self.features = df[feature_cols].values.astype('float32')
        self.targets = df[target_cols].values.astype('float32')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y