import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import threading

class SingleViewDataset(Dataset):
    def __init__(self, hdf_path, target_names, indices, transform=None, target_transform=None, task_type='regression', class_to_idx=None):
        self.hdf_path = hdf_path
        self.target_names = target_names if isinstance(target_names, list) else [target_names]
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self._thread_local = threading.local()
        self.task_type = task_type
        if class_to_idx is None:
            with h5py.File(self.hdf_path, 'r') as ds_file:
                # Read the entire target array once
                all_targets = ds_file[self.target_names[0]][:]
            unique_targets = sorted(set(float(t) for t in all_targets))
            self.class_to_idx = {f"{t:.1f}": idx for idx, t in enumerate(unique_targets)}
        else:
            self.class_to_idx = class_to_idx

    def _get_file(self):
        if not hasattr(self._thread_local, "ds_file"):
            self._thread_local.ds_file = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True, locking=False)
        return self._thread_local.ds_file

    def __getitem__(self, idx):
        # Ensure idx is a Python int (not a tensor)
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        ds_file = self._get_file()
        real_idx = self.indices[idx]
        img = ds_file['images'][real_idx]  # shape (H, W)
        # img_3chan = np.repeat(img[None, :, :], 3, axis=0).astype(np.float32)  # (3, H, W)
        img = np.repeat(img[None, :, :], 1, axis=0).astype(np.float32)  # (1, H, W)
        img_tensor = torch.from_numpy(img)  # Convert to tensor
        targets = [ds_file[name][real_idx] for name in self.target_names]

        # Map targets to class indices
        if self.task_type == 'classification':
            target = self.class_to_idx[f"{float(targets[0]):.1f}"]
            target_tensor = torch.tensor(target, dtype=torch.long)

        # Apply input transform (should be tensor transforms only)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Apply output transform (e.g., log-transform)
        if self.task_type == 'regression':
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            if self.target_transform:
                target_tensor = self.target_transform(target_tensor)

        return img_tensor, target_tensor

    def __len__(self):
        return len(self.indices)