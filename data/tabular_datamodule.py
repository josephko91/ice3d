import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .tabular_dataset import TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file,
        feature_names,
        target_names,
        batch_size=32,
        subset_size=1.0,
        subset_seed=42,
        num_workers=4,
        task_type='regression',
        class_to_idx=None,
        target_transform=None,
        train_idx=None,
        val_idx=None,
        test_idx=None,
    ):
        super().__init__()
        self.data_file = data_file
        self.feature_names = feature_names
        self.target_names = target_names
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.num_workers = num_workers
        self.task_type = task_type
        self.class_to_idx = class_to_idx
        self.target_transform = target_transform
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    def setup(self, stage=None):
        # Load the full DataFrame
        if self.data_file.endswith('.csv'):
            df = pd.read_csv(self.data_file)
        elif self.data_file.endswith('.parquet'):
            df = pd.read_parquet(self.data_file)
        else:
            raise ValueError(f"Unsupported file type: {self.data_file}")

        # Subset before splitting, if needed
        if self.subset_size < 1.0:
            df = df.sample(frac=self.subset_size, random_state=self.subset_seed).reset_index(drop=True)

        # Infer feature columns if not provided
        if self.feature_names is None:
            feature_cols = [col for col in df.columns if col not in self.target_names]
        else:
            feature_cols = self.feature_names
        target_cols = self.target_names

        # If indices are not provided, generate them here
        if (self.train_idx is None) or (self.val_idx is None) or (self.test_idx is None):
            indices = list(range(len(df)))
            train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=self.subset_seed)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=self.subset_seed)
        else:
            def read_indices(idx):
                if isinstance(idx, str):
                    with open(idx, 'r') as f:
                        return [int(line.strip()) for line in f if line.strip()]
                return idx
            train_idx = read_indices(self.train_idx)
            val_idx = read_indices(self.val_idx)
            test_idx = read_indices(self.test_idx)

        # Create datasets for each split using the indices
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

        # Fit scaler on the training features
        train_features = train_df[self.feature_names].values.astype('float32')
        self.scaler = StandardScaler().fit(train_features)
        # Define transform function for use in TabularDataset
        def std_scale(x):
            # x is a 1D torch tensor
            x_np = x.numpy().reshape(1, -1)
            x_scaled = self.scaler.transform(x_np)[0]
            return torch.from_numpy(x_scaled.astype('float32'))
        
        # Build class_to_idx if needed
        if self.class_to_idx is None and self.task_type == 'classification':
            if not isinstance(self.target_names, (list, tuple)) or len(self.target_names) == 0:
                raise ValueError("target_names must be a non-empty list or tuple.")
            unique_targets = sorted(train_df[self.target_names[0]].unique())
            self.class_to_idx = {str(t): idx for idx, t in enumerate(unique_targets)}

        self.train_set = TabularDataset(
            train_df, feature_cols, target_cols, 
            task_type=self.task_type, class_to_idx=self.class_to_idx,
            transform=std_scale, target_transform=self.target_transform
        )
        self.val_set = TabularDataset(
            val_df, feature_cols, target_cols, 
            task_type=self.task_type, class_to_idx=self.class_to_idx,
            transform=std_scale, target_transform=self.target_transform
        )
        self.test_set = TabularDataset(
            test_df, feature_cols, target_cols, 
            task_type=self.task_type, class_to_idx=self.class_to_idx,
            transform=std_scale, target_transform=self.target_transform
        )

        self.input_size = len(feature_cols)
        self.num_classes = len(self.class_to_idx) if self.task_type == 'classification' else None

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)