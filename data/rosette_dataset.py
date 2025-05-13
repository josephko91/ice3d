import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import numpy as np

class CombinedHDF5Dataset(Dataset):
    def __init__(self, default_path, ds2_path):
        self.default_file = h5py.File(default_path, 'r')
        self.ds2_file = h5py.File(ds2_path, 'r')
        
        self.default_data = self.default_file['images']
        self.ds2_data = self.ds2_file['images']
        
        # Check matching dimensions
        if len(self.default_data) != len(self.ds2_data):
            raise ValueError("Datasets must have the same number of samples")
        
        self.length = len(self.default_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        default_img = self.default_data[idx]     # shape (224, 224)
        ds2_img = self.ds2_data[idx]             # shape (224, 224)

        # Stack channels: (3, 224, 224)
        combined = np.stack([
            default_img,
            ds2_img,
            ds2_img
        ], axis=0).astype(np.float32)

        return torch.from_numpy(combined)

    def __del__(self):
        # Cleanly close files
        self.default_file.close()
        self.ds2_file.close()


class RosetteDataSet(Dataset):
    def __init__(self, image_dir, labels_file, split='train', transform=None, target='n_arms'):
        """
        Args:
            image_dir (string): Directory with subfolders ('train', 'val', 'test') containing images.
            csv_file (string): CSV file containing image filenames and labels.
            split (string): Which data split to use ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
            target (string): ['n_arms', 'rho_eff', 'sa_eff']
        """
        self.image_dir = os.path.join(image_dir, split)  # Select the correct subfolder (train/val/test)
        self.labels = pd.read_csv(labels_file)
        self.labels = self.labels[self.labels['split'] == split].reset_index(drop=True)
        self.label_mapping = {
            '4': 0,
            '5': 1,
            '6': 2,
            '7': 3,
            '8': 4,
            '9': 5,
            '10': 6
        }
        self.transform = transform
        self.target = target

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.labels)

    def __getitem__(self, idx):
        """Load an image and its corresponding label(s)"""
        row = self.labels.iloc[idx]
        img_name = row['filename']  # Get the image filename
        img_path = os.path.join(self.image_dir, img_name)  # Construct the full image path
        image = Image.open(img_path)
        # Classification label (e.g. n_arms)
        label_str = str(row['n_arms'])  # ensure it's a string
        n_arms_labels = torch.tensor(self.label_mapping[label_str], dtype=torch.long)
        # Regression targets
        regression_targets = torch.tensor([row['rho_eff'], row['sa_eff']], dtype=torch.float32)
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        return image, n_arms_labels, regression_targets