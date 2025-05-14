import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from .single_view_dataset import SingleViewDataset

class SingleViewDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hdf_file,
        target_names,
        train_idx,
        val_idx,
        test_idx,
        batch_size=32,
        subset_size=None,
        subset_seed=42,
        num_workers=4,
        prefetch_factor=4,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        train_target_transform=None,
        val_target_transform=None,
        test_target_transform=None,
        task_type='regression',
        class_to_idx=None
    ):
        super().__init__()
        self.target_names = [target_names] if isinstance(target_names, str) else target_names
        self.hdf_file = hdf_file
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_target_transform = train_target_transform
        self.val_target_transform = val_target_transform
        self.test_target_transform = test_target_transform
        self.task_type = task_type
        self.class_to_idx = class_to_idx

    def _subset_indices(self, indices):
        if self.subset_size is not None and 0 < self.subset_size < 1:
            rng = np.random.default_rng(self.subset_seed)
            subset_count = max(1, int(len(indices) * self.subset_size))
            indices = rng.choice(indices, size=subset_count, replace=False)
        indices = np.sort(indices)
        return indices

    def setup(self, stage=None):
        train_idx = self._subset_indices(self.train_idx)
        val_idx = self._subset_indices(self.val_idx)
        test_idx = self._subset_indices(self.test_idx)
        self.train_dataset = SingleViewDataset(
            self.hdf_file, self.target_names, train_idx,
            transform=self.train_transform,
            target_transform=self.train_target_transform,
            task_type=self.task_type,
            class_to_idx=self.class_to_idx
        )
        self.val_dataset = SingleViewDataset(
            self.hdf_file, self.target_names, val_idx,
            transform=self.val_transform,
            target_transform=self.val_target_transform,
            task_type=self.task_type,
            class_to_idx=self.class_to_idx
        )
        self.test_dataset = SingleViewDataset(
            self.hdf_file, self.target_names, test_idx,
            transform=self.test_transform,
            target_transform=self.test_target_transform,
            task_type=self.task_type,
            class_to_idx=self.class_to_idx
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor
        )