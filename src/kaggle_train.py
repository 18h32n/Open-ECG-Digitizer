"""
Simplified training script for Kaggle ECG competition.

This script is designed for:
- Running on Kaggle notebooks (P100/T4 GPUs)
- End-to-end training with signal reconstruction loss
- Simple checkpointing without Ray Tune dependency
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.default import get_cfg, merge_cfg
from src.dataset.kaggle_ecg import KaggleECGDataset
from src.loss.signal_loss import SignalReconstructionLoss
from src.model.unet import UNet
from src.model.signal_head import SimpleDifferentiableSignalHead, UNetWithSignalHead
from src.utils import find_config_path


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(
    unet_weights_path: Optional[str] = None,
    signal_length: int = 5000,
    freeze_unet: bool = False,
    device: torch.device = torch.device("cpu"),
) -> UNetWithSignalHead:
    """
    Create the UNet + Signal Head model.

    Args:
        unet_weights_path: Path to pre-trained UNet weights
        signal_length: Target signal length
        freeze_unet: Whether to freeze UNet weights
        device: Target device

    Returns:
        Combined model
    """
    # Create UNet (standard config from unet.yml)
    unet = UNet(
        num_in_channels=3,
        num_out_channels=4,
        depth=3,
        dims=[32, 64, 128, 256, 512],
    )

    # Load pre-trained weights if available
    if unet_weights_path and os.path.exists(unet_weights_path):
        print(f"Loading UNet weights from {unet_weights_path}")
        state_dict = torch.load(unet_weights_path, map_location=device)
        unet.load_state_dict(state_dict, strict=False)

    # Create signal head
    signal_head = SimpleDifferentiableSignalHead(
        signal_length=signal_length,
        num_leads=12,
        signal_class=2,
    )

    # Combine into end-to-end model
    model = UNetWithSignalHead(
        unet=unet,
        signal_head=signal_head,
        signal_length=signal_length,
        freeze_unet=freeze_unet,
    )

    return model


def create_dataloaders(
    train_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    signal_length: int = 5000,
    val_ratio: float = 0.1,
    degradation_types: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_dir: Path to training data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        signal_length: Target signal length
        val_ratio: Validation split ratio
        degradation_types: Which degradation types to include

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = KaggleECGDataset(
        train_dir=train_dir,
        transform=None,  # TODO: Add augmentations
        target_length=signal_length,
        degradation_types=degradation_types,
        cache_signals=True,
        split="train",
        val_ratio=val_ratio,
    )

    val_dataset = KaggleECGDataset(
        train_dir=train_dir,
        transform=None,
        target_length=signal_length,
        degradation_types=degradation_types,
        cache_signals=True,
        split="val",
        val_ratio=val_ratio,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")

    for batch in progress_bar:
        # Unpack batch
        images, signals, lead_mask, metadata = batch
        images = images.to(device)
        signals = signals.to(device)
        lead_mask = lead_mask.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.autocast(device_type=str(device.type), enabled=scaler is not None):
            seg_output, pred_signals, pred_mask = model(images)
            loss = criterion(pred_signals, signals, lead_mask)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return {
        "train_loss": total_loss / max(num_batches, 1),
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val]")

    for batch in progress_bar:
        images, signals, lead_mask, metadata = batch
        images = images.to(device)
        signals = signals.to(device)
        lead_mask = lead_mask.to(device)

        # Forward pass (no autocast needed for validation)
        seg_output, pred_signals, pred_mask = model(images)
        loss = criterion(pred_signals, signals, lead_mask)

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return {
        "val_loss": total_loss / max(num_batches, 1),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def train(
    train_dir: str,
    output_dir: str = "/kaggle/working",
    unet_weights_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    signal_length: int = 5000,
    freeze_unet_epochs: int = 5,
    use_amp: bool = True,
    checkpoint_freq: int = 1,
    resume_from: Optional[str] = None,
    degradation_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        train_dir: Path to training data directory
        output_dir: Directory to save checkpoints
        unet_weights_path: Path to pre-trained UNet weights
        epochs: Total number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        signal_length: Target signal length
        freeze_unet_epochs: Number of epochs to freeze UNet
        use_amp: Use automatic mixed precision
        checkpoint_freq: Save checkpoint every N epochs
        resume_from: Path to checkpoint to resume from
        degradation_types: Which degradation types to train on

    Returns:
        Dictionary with training history
    """
    device = get_device()
    print(f"Using device: {device}")

    # Create model
    model = create_model(
        unet_weights_path=unet_weights_path,
        signal_length=signal_length,
        freeze_unet=True,  # Start with frozen UNet
        device=device,
    )
    model = model.to(device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        batch_size=batch_size,
        signal_length=signal_length,
        degradation_types=degradation_types,
    )

    # Create loss function
    criterion = SignalReconstructionLoss(
        mse_weight=1.0,
        correlation_weight=0.5,
        goldberger_weight=0.1,
    )

    # Create optimizer (only signal head parameters initially)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=learning_rate / 10,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, resume_from, device)
        start_epoch += 1  # Start from next epoch

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    print(f"Starting training for {epochs} epochs...")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(start_epoch, epochs):
        # Unfreeze UNet after freeze_unet_epochs
        if epoch == freeze_unet_epochs and freeze_unet_epochs > 0:
            print(f"\n=== Unfreezing UNet at epoch {epoch + 1} ===")
            for param in model.unet.parameters():
                param.requires_grad = True

            # Recreate optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate / 10,  # Lower LR for fine-tuning
                weight_decay=weight_decay,
            )

            # Recreate scheduler
            remaining_epochs = epochs - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs * len(train_loader),
                eta_min=learning_rate / 100,
            )

            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Train epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            max_epochs=epochs,
        )

        # Update scheduler
        scheduler.step()

        # Validate epoch
        val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_epochs=epochs,
        )

        # Update history
        history["train_loss"].append(train_metrics["train_loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["lr"].append(scheduler.get_last_lr()[0])

        # Print epoch summary
        print(
            f"\nEpoch {epoch + 1}/{epochs}: "
            f"Train Loss = {train_metrics['train_loss']:.4f}, "
            f"Val Loss = {val_metrics['val_loss']:.4f}, "
            f"LR = {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_dir=output_dir,
                filename="best_model.pt",
            )
            print(f"Saved best model (val_loss = {best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_dir=output_dir,
                filename=f"checkpoint_epoch_{epoch + 1}.pt",
            )

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs - 1,
        best_val_loss=best_val_loss,
        checkpoint_dir=output_dir,
        filename="final_model.pt",
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ECG digitization model")
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--unet-weights",
        type=str,
        default=None,
        help="Path to pre-trained UNet weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=5,
        help="Number of epochs to freeze UNet",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        unet_weights_path=args.unet_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_unet_epochs=args.freeze_epochs,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
