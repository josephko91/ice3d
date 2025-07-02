import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18

class ResNet18Classification(pl.LightningModule):
    def __init__(self, input_channels, num_classes, 
                 learning_rate=1e-3, weights=False):
        super(ResNet18Classification, self).__init__()
        self.learning_rate = learning_rate

        # Load ResNet-18
        self.resnet18 = resnet18(weights=weights)

        # Modify the first convolutional layer to accept the specified input_channels
        if input_channels != 3:
            self.resnet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the fully connected layer to output the specified number of classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
