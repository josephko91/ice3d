import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .rosette_dataset import RosetteDataSet

class RosetteDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, labels_file, batch_size, target, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.batch_size = batch_size
        self.target = target
        self.transform = transform

    def setup(self, stage=None):
        self.train_dataset = RosetteDataSet(self.image_dir, self.labels_file, split='train', transform=self.transform, target=self.target)
        self.val_dataset = RosetteDataSet(self.image_dir, self.labels_file, split='val', transform=self.transform, target=self.target)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)