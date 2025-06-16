import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')  # <-- Add this line at the very top, before importing pyplot
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchvision.transforms as T
import json

# Add your project root to sys.path for imports
sys.path.append('/home/jko/ice3d')

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
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    
def get_transforms(args):
    transforms = {}
    # Define transforms based on data_type
    if args.data_type in ['single_view_h5', 'stereo_view_h5']:
        train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.Normalize(mean=[0.5] * args.input_channels, std=[1.0] * args.input_channels)
            ])
        val_transform = T.Compose([
                T.Normalize(mean=[0.5] * args.input_channels, std=[1.0] * args.input_channels)
            ])
        transforms['train'] = train_transform
        transforms['val'] = val_transform
        transforms['test'] = val_transform
        # define target transform
        if args.task_type == 'classification':
            target_transform = None
        else:
            def log_transform(x):
                return torch.log(x)
            target_transform = log_transform
        transforms['train_target'] = target_transform
        transforms['val_target'] = target_transform
        transforms['test_target'] = target_transform    
        return transforms
    elif args.data_type == 'tabular':
        # define target transform
        if args.task_type == 'classification':
            target_transform = None
        else:
            def log_transform(x):
                return torch.log(x)
            target_transform = log_transform
        transforms['target'] = target_transform
        return transforms
    else:
        return None

def get_datamodule(args, class_to_idx=None):
    transforms = get_transforms(args)
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
            train_transform=transforms['train'],
            val_transform=transforms['val'],
            test_transform=transforms['test'],
            train_target_transform=transforms['train_target'],
            val_target_transform=transforms['val_target'],
            test_target_transform=transforms['test_target'],
            task_type=args.task_type,
            class_to_idx=class_to_idx
        )
    elif args.data_type == 'stereo_view_h5':
        return StereoViewDataModule(
            hdf_file_left=args.hdf_file_left,
            hdf_file_right=args.hdf_file_right,
            target_names=args.targets.split(','),
            train_idx=None,
            val_idx=None,
            test_idx=None,
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            subset_seed=args.seed,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            train_transform=transforms['train'],
            val_transform=transforms['val'],
            test_transform=transforms['test'],
            train_target_transform=transforms['train_target'],
            val_target_transform=transforms['val_target'],
            test_target_transform=transforms['test_target'],
            task_type=args.task_type,
            class_to_idx=class_to_idx
        )
    elif args.data_type == 'tabular':
        feature_names = args.feature_names.split(',') if args.feature_names else None
        return TabularDataModule(
            data_file=args.tabular_file,
            feature_names=feature_names,
            target_names=args.targets.split(','),
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            subset_seed=args.seed,
            num_workers=args.num_workers,
            task_type=args.task_type,
            class_to_idx=class_to_idx,
            target_transform=transforms['target'],
            train_idx=args.train_idx,
            val_idx=args.val_idx,   
            test_idx=args.test_idx
        )
    else:
        raise ValueError(f"Unknown data type: {args.data_type}")

@rank_zero_only
def evaluate_and_plot(model, dm, args):
    print("Evaluating model...")
    # Put model in eval mode
    # model.eval()

    # Get validation dataloader
    val_loader = dm.val_dataloader()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch['x'], batch['y']
            x = x.to(model.device)
            y = y.to(model.device)
            outputs = model(x)
            if args.task_type == 'classification':
                if outputs.shape[-1] > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (outputs > 0.5).long().squeeze()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
            else:
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    fig_save_path = os.path.join(args.log_dir, f"{args.model}_{args.data_type}_val_plot.png")

    if args.task_type == 'classification':
        cm = confusion_matrix(all_targets.flatten(), all_preds.flatten())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(fig_save_path)
        plt.close()
    else:
        num_targets = all_targets.shape[1] if all_targets.ndim > 1 else 1
        plt.figure(figsize=(6 * num_targets, 6))
        for i in range(num_targets):
            plt.subplot(1, num_targets, i + 1)
            t = all_targets[:, i] if num_targets > 1 else all_targets
            p = all_preds[:, i] if num_targets > 1 else all_preds
            plt.scatter(t.flatten(), p.flatten(), alpha=0.5)
            plt.xlabel(f"True Values (Target {i})")
            plt.ylabel(f"Predicted Values (Target {i})")
            plt.title(f"Validation Scatter Plot (Target {i})")
            min_val = min(t.min(), p.min())
            max_val = max(t.max(), p.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.tight_layout()
        plt.savefig(fig_save_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch models with flexible options.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['mlp_regression', 'mlp_classification', 'cnn_regression', 'cnn_classification', 'resnet18_regression', 'resnet18_classification'])
    parser.add_argument('--data_type', type=str, required=True,
                        choices=['single_view_h5', 'stereo_view_h5', 'tabular'])
    parser.add_argument('--hdf_file', type=str, default=None, help='Path to HDF5 file (for single view)')
    parser.add_argument('--hdf_file_left', type=str, default=None, help='Path to left HDF5 file (for stereo view)')
    parser.add_argument('--hdf_file_right', type=str, default=None, help='Path to right HDF5 file (for stereo view)')
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
    parser.add_argument('--class_to_idx_json', type=str, default=None, help='Path to class_to_idx JSON file')
    parser.add_argument('--feature_names', type=str, default=None, help='Comma-separated list of feature column names')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--train_idx', type=str, default=None, help='Path to training indices file')
    parser.add_argument('--val_idx', type=str, default=None, help='Path to validation indices file')
    parser.add_argument('--test_idx', type=str, default=None, help='Path to test indices file')
    args = parser.parse_args()

    # Load class_to_idx mapping if provided
    class_to_idx = None
    if args.class_to_idx_json is not None:
        with open(args.class_to_idx_json, 'r') as f:
            class_to_idx = json.load(f)
    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(args.log_dir, name=args.tb_log_name)
    csv_logger = CSVLogger(args.log_dir, name=args.csv_log_name)

    dm = get_datamodule(args, class_to_idx=class_to_idx)
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
        if class_to_idx is not None:
            num_classes = len(class_to_idx)
        else: # default to 7 classes
            num_classes = 7
    elif args.model.endswith('regression'):
        output_size = len(args.targets.split(','))

    model = get_model(args, input_size=input_size, output_size=output_size, num_classes=num_classes)

    # early_stop_callback = EarlyStopping(
    # monitor="val_loss",      # Metric to monitor
    # mode="min",              # "min" for loss, "max" for accuracy
    # patience=5,              # Stop if no improvement after N validations
    # min_delta=0.001,         # Minimum change to qualify as improvement
    # verbose=True             # Print messages on early stopping
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",         # Metric to monitor
        mode="min",                 # Save checkpoints with lower val_loss
        save_top_k=3,               # Save the 3 best models
        filename="model-{epoch:02d}-{val_loss:.4f}",  # Custom filename
        # every_n_epochs=1,           # Save every epoch (optional)
        save_last=True              # Also save the last epoch
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        # devices=args.num_gpus,
        # strategy="ddp",
        logger=[csv_logger, tb_logger],
        enable_progress_bar=True,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()