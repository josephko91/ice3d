import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import threading

class SingleViewDataset(Dataset):
    def __init__(self, hdf_path, target_names, indices, transform=None, target_transform=None):
        self.hdf_path = hdf_path
        self.target_names = target_names if isinstance(target_names, list) else [target_names]
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self._thread_local = threading.local()

    def _get_file(self):
        if not hasattr(self._thread_local, "ds_file"):
            self._thread_local.ds_file = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True, locking=False)
        return self._thread_local.ds_file

    def __getitem__(self, idx):
        ds_file = self._get_file()
        real_idx = self.indices[idx]
        img = ds_file['images'][real_idx]  # shape (H, W)
        img_3chan = np.repeat(img[None, :, :], 3, axis=0).astype(np.float32)  # (3, H, W)
        img_tensor = torch.from_numpy(img_3chan)  # Convert to tensor here
        targets = [ds_file[name][real_idx] for name in self.target_names]
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        # Apply input transform (should be tensor transforms only)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Apply output transform (e.g., log-transform)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)

        return img_tensor, target_tensor

    def __len__(self):
        return len(self.indices)