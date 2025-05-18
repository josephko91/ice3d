import pandas as pd
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from .tabular_dataset import TabularDataset

class TabularDataModule(pl.LightningDataModule):
    def __init__(self, data_file, target_names, batch_size=32, subset_size=1.0, subset_seed=42, num_workers=4, task_type='regression'):
        super().__init__()
        self.data_file = data_file
        self.target_names = target_names
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.num_workers = num_workers
        self.task_type = task_type

    def setup(self, stage=None):
        df = pd.read_csv(self.data_file)
        feature_cols = [col for col in df.columns if col not in self.target_names]
        target_cols = self.target_names

        if self.subset_size < 1.0:
            df = df.sample(frac=self.subset_size, random_state=self.subset_seed)

        dataset = TabularDataset(df, feature_cols, target_cols)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(self.subset_seed)
        )

        self.input_size = len(feature_cols)
        self.num_classes = len(df[target_cols[0]].unique()) if self.task_type == 'classification' else None

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)