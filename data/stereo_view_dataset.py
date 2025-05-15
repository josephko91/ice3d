import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import threading

class StereoViewDataset(Dataset):
    def __init__(self, hdf_path1, hdf_path2, target_names, indices, transform=None, target_transform=None, task_type='regression', class_to_idx=None):
        self.hdf_path1 = hdf_path1
        self.hdf_path2 = hdf_path2
        self.target_names = target_names if isinstance(target_names, list) else [target_names]
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self._thread_local = threading.local()
        self.task_type = task_type
        if class_to_idx is None:
            with h5py.File(self.hdf_path1, 'r') as ds_file:
                # Read the entire target array once
                all_targets = ds_file[self.target_names[0]][:]
            unique_targets = sorted(set(float(t) for t in all_targets))
            self.class_to_idx = {f"{t:.1f}": idx for idx, t in enumerate(unique_targets)}
        else:
            self.class_to_idx = class_to_idx

    def _get_file(self, path_key):
        if not hasattr(self._thread_local, path_key):
            setattr(self._thread_local, path_key, h5py.File(getattr(self, path_key), 'r', libver='latest', swmr=True, locking=False))
        return getattr(self._thread_local, path_key)

    def __getitem__(self, idx):
        ds_file1 = self._get_file('hdf_path1')
        ds_file2 = self._get_file('hdf_path2')
        real_idx = self.indices[idx]

        # Load single-channel images from both files
        img1 = ds_file1['images'][real_idx]  # shape (H, W)
        img2 = ds_file2['images'][real_idx]  # shape (H, W)

        # Combine into a 2-channel image
        img_2chan = np.stack([img1, img2], axis=0).astype(np.float32)  # (2, H, W)
        img_tensor = torch.from_numpy(img_2chan)  # Convert to tensor here

        targets = [ds_file1[name][real_idx] for name in self.target_names]

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