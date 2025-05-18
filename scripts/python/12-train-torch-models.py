import argparse
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Add your project root to sys.path for imports
sys.path.append('/glade/u/home/joko/ice3d')

# Import your models and datamodules
from models.mlp_regression import MLPRegression
from models.mlp_classification import MLPClassification
from models.cnn_regression import VanillaCNNRegression
from models.cnn_classification import VanillaCNNClassification
from models.resnet18_regression import ResNet18Regression
from models.resnet18_classification import ResNet18Classification
from data.single_view_datamodule import SingleViewDataModule
from data.stereo_view_datamodule import StereoViewDataModule
from data.tabular_datamodule import TabularDataModule

def get_model(args, input_size=None, output_size=None, num_classes=None):
    if args.model == 'mlp_regression':
        return MLPRegression(input_size, output_size, learning_rate=args.lr)
    elif args.model == 'mlp_classification':
        return MLPClassification(input_size, num_classes, learning_rate=args.lr)
    elif args.model == 'cnn_regression':
        return VanillaCNNRegression(input_channels=args.input_channels, output_size=output_size, learning_rate=args.lr)
    elif args.model == 'cnn_classification':
        return VanillaCNNClassification(input_channels=args.input_channels, num_classes=num_classes, learning_rate=args.lr)
    elif args.model == 'resnet18_regression':
        return ResNet18Regression(input_channels=args.input_channels, output_size=output_size, learning_rate=args.lr)
    elif args.model == 'resnet18_classification':
        return ResNet18Classification(input_channels=args.input_channels, num_classes=num_classes, learning_rate=args.lr)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

def get_datamodule(args):
    if args.data_type == 'single_view_h5':
        return SingleViewDataModule(
            hdf_file=args.hdf_file,
            target_names=args.targets.split(','),
            train_idx=None,
            val_idx=None,
            test_idx=None,
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            subset_seed=args.seed,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            train_transform=None,
            val_transform=None,
            test_transform=None,
            train_target_transform=None,
            val_target_transform=None,
            test_target_transform=None,
            task_type=args.task_type
        )
    elif args.data_type == 'stereo_view_h5':
        return StereoViewDataModule(
            hdf_file=args.hdf_file,
            target_names=args.targets.split(','),
            train_idx=None,
            val_idx=None,
            test_idx=None,
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            subset_seed=args.seed,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            train_transform=None,
            val_transform=None,
            test_transform=None,
            train_target_transform=None,
            val_target_transform=None,
            test_target_transform=None,
            task_type=args.task_type
        )
    elif args.data_type == 'tabular':
        return TabularDataModule(
            data_file=args.tabular_file,
            target_names=args.targets.split(','),
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            subset_seed=args.seed,
            num_workers=args.num_workers,
            task_type=args.task_type
        )
    else:
        raise ValueError(f"Unknown data type: {args.data_type}")

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch models with flexible options.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['mlp_regression', 'mlp_classification', 'cnn_regression', 'cnn_classification', 'resnet18_regression', 'resnet18_classification'])
    parser.add_argument('--data_type', type=str, required=True,
                        choices=['single_view_h5', 'two_view_h5', 'tabular'])
    parser.add_argument('--hdf_file', type=str, default=None, help='Path to HDF5 file (for h5 data)')
    parser.add_argument('--tabular_file', type=str, default=None, help='Path to tabular data file (CSV/Parquet)')
    parser.add_argument('--targets', type=str, default='rho_eff,sa_eff', help='Comma-separated list of target names')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels for image models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--subset_size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--task_type', type=str, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--log_dir', type=str, default='./lightning_logs')
    parser.add_argument('--tb_log_name', type=str, default='tb')
    parser.add_argument('--csv_log_name', type=str, default='csv')
    args = parser.parse_args()

    tb_logger = TensorBoardLogger(args.log_dir, name=args.tb_log_name)
    csv_logger = CSVLogger(args.log_dir, name=args.csv_log_name)

    dm = get_datamodule(args)
    dm.setup()

    input_size = None
    output_size = None
    num_classes = None
    if args.model.startswith('mlp'):
        # For tabular data, infer input/output sizes from datamodule
        if args.data_type == 'tabular':
            input_size = dm.input_size
            if args.task_type == 'regression':
                output_size = len(args.targets.split(','))
            else:
                num_classes = dm.num_classes
    elif args.model.endswith('classification'):
        # For image classification, get num_classes from datamodule or user
        num_classes = getattr(dm, 'num_classes', args.input_channels)  # fallback to input_channels

    model = get_model(args, input_size=input_size, output_size=output_size, num_classes=num_classes)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=[csv_logger, tb_logger],
        enable_progress_bar=True,
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()