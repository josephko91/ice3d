import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(
            self, df, 
            feature_cols, 
            target_cols, 
            task_type='regression', 
            class_to_idx=None,
            transform=None,
            target_transform=None
        ):
        self.features = df[feature_cols].values.astype('float32')
        self.targets_raw = df[target_cols].values
        self.task_type = task_type
        self.target_cols = target_cols
        self.transform = transform
        self.target_transform = target_transform
        if self.task_type == 'classification':
            if class_to_idx is None:
                unique_targets = sorted(set(float(t) for t in self.targets_raw[:, 0]))
                self.class_to_idx = {f"{t:.1f}": idx for idx, t in enumerate(unique_targets)}
            else:
                self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        if self.transform:
            x = self.transform(x)
        if self.task_type == 'classification':
            # Map the first target column to class index
            target_val = self.targets_raw[idx][0]
            if self.class_to_idx:
                target_idx = self.class_to_idx[str(target_val)]
            else:
                target_idx = int(target_val)
            y = torch.tensor(target_idx, dtype=torch.long)
        else:
            y = torch.tensor(self.targets_raw[idx].astype('float32'))
        if self.target_transform:
            y = self.target_transform(y)
        return x, y