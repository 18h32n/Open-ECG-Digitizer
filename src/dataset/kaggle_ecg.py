"""
Dataset for Kaggle PhysioNet ECG Image Digitization competition.

Loads ECG images paired with ground truth time-series CSV files.
Handles the competition's specific data structure with 8 degradation
variants per sample and variable-length lead data.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal
from torchvision.io import decode_image
import torchvision.transforms.functional as TF


# Standard 12-lead order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Degradation suffixes in competition data
DEGRADATION_SUFFIXES = [
    "0001",  # Original clean
    "0003",  # Color printed, color scanned
    "0004",  # Color printed, B&W scanned
    "0005",  # Mobile photo of print
    "0006",  # Mobile photo of screen
    "0009",  # Stained/soaked
    "0010",  # Extensive damage
    "0011",  # Moldy (color)
    "0012",  # Moldy (B&W)
]


class KaggleECGDataset(torch.utils.data.Dataset[Any]):
    """
    Dataset for Kaggle ECG Image Digitization competition.

    Expected structure:
        train_dir/[id]/[id].csv (ground truth signals)
        train_dir/[id]/[id]-XXXX.png (degraded images)

    Returns:
        image: Tensor (3, H, W) normalized to [0, 1]
        signals: Tensor (12, target_length) in mV
        lead_mask: Tensor (12, target_length) binary mask for valid samples
        metadata: dict with sample_id, degradation_type, fs
    """

    def __init__(
        self,
        train_dir: str,
        transform: Any = None,
        target_length: int = 5000,
        target_fs: float = 500.0,
        degradation_types: Optional[List[str]] = None,
        cache_signals: bool = True,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            train_dir: Path to competition train directory
            transform: Optional transform to apply to images
            target_length: Target signal length after resampling
            target_fs: Target sampling frequency
            degradation_types: List of degradation suffixes to include (default: all)
            cache_signals: Whether to cache loaded signal CSVs
            split: "train" or "val" for train/validation split
            val_ratio: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.train_dir = train_dir
        self.transform = transform
        self.target_length = target_length
        self.target_fs = target_fs
        self.degradation_types = degradation_types or DEGRADATION_SUFFIXES
        self.cache_signals = cache_signals

        # Find all sample IDs
        self.sample_ids = self._find_sample_ids()

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.sample_ids))
        val_size = int(len(self.sample_ids) * val_ratio)

        if split == "val":
            self.sample_ids = [self.sample_ids[i] for i in indices[:val_size]]
        else:  # train
            self.sample_ids = [self.sample_ids[i] for i in indices[val_size:]]

        # Build index of (sample_id, degradation_suffix) pairs
        self.image_index = self._build_image_index()

        # Signal cache
        self._signal_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _find_sample_ids(self) -> List[str]:
        """Find all sample IDs in the train directory."""
        if not os.path.exists(self.train_dir):
            raise ValueError(f"Train directory does not exist: {self.train_dir}")

        sample_ids = []
        for name in os.listdir(self.train_dir):
            sample_dir = os.path.join(self.train_dir, name)
            if os.path.isdir(sample_dir):
                # Check that CSV exists
                csv_path = os.path.join(sample_dir, f"{name}.csv")
                if os.path.exists(csv_path):
                    sample_ids.append(name)

        return sorted(sample_ids)

    def _build_image_index(self) -> List[Tuple[str, str]]:
        """Build index of (sample_id, degradation_suffix) pairs."""
        index = []
        for sample_id in self.sample_ids:
            sample_dir = os.path.join(self.train_dir, sample_id)
            for suffix in self.degradation_types:
                img_path = os.path.join(sample_dir, f"{sample_id}-{suffix}.png")
                if os.path.exists(img_path):
                    index.append((sample_id, suffix))
        return index

    def _load_signals(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load ground truth signals from CSV.

        The competition CSV format has:
        - Lead II: Full 10 seconds (all rows)
        - Other leads: 2.5 seconds each in different row ranges

        Returns:
            signals: Tensor (12, target_length) in mV
            lead_mask: Tensor (12, target_length) binary mask
        """
        # Check cache
        if self.cache_signals and sample_id in self._signal_cache:
            return self._signal_cache[sample_id]

        csv_path = os.path.join(self.train_dir, sample_id, f"{sample_id}.csv")
        df = pd.read_csv(csv_path)

        # Initialize output tensors
        signals = torch.zeros(12, self.target_length)
        lead_mask = torch.zeros(12, self.target_length)

        # Get original sampling frequency (10000 samples / 10 seconds = 1000 Hz typical)
        n_total_samples = len(df)
        original_fs = n_total_samples / 10.0  # 10 seconds total duration

        for i, lead_name in enumerate(LEAD_NAMES):
            if lead_name not in df.columns:
                continue

            values = df[lead_name].values
            valid_mask = ~np.isnan(values)

            if not valid_mask.any():
                continue

            # Extract valid values
            valid_indices = np.where(valid_mask)[0]
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1] + 1
            valid_values = values[first_valid:last_valid]

            # Handle any internal NaNs with interpolation
            if np.isnan(valid_values).any():
                nan_mask = np.isnan(valid_values)
                valid_values = np.interp(
                    np.arange(len(valid_values)),
                    np.where(~nan_mask)[0],
                    valid_values[~nan_mask]
                )

            # Determine target length for this lead
            if lead_name == "II":
                # Lead II has full 10 seconds
                lead_target_length = self.target_length
            else:
                # Other leads have 2.5 seconds
                lead_target_length = self.target_length // 4

            # Resample to target length
            if len(valid_values) != lead_target_length:
                resampled = scipy_signal.resample(valid_values, lead_target_length)
            else:
                resampled = valid_values

            # Store in output tensors
            signals[i, :lead_target_length] = torch.from_numpy(resampled.astype(np.float32))
            lead_mask[i, :lead_target_length] = 1.0

        result = (signals, lead_mask)

        # Cache result
        if self.cache_signals:
            self._signal_cache[sample_id] = result

        return result

    def _load_image(self, sample_id: str, degradation_suffix: str) -> torch.Tensor:
        """Load ECG image as RGB tensor normalized to [0, 1]."""
        img_path = os.path.join(
            self.train_dir, sample_id, f"{sample_id}-{degradation_suffix}.png"
        )
        image = decode_image(img_path, mode="RGB").float() / 255.0
        # Resize to consistent dimensions for batching (H=1700, W=2200)
        image = TF.resize(image, size=[1700, 2200], antialias=True)
        return image

    def __len__(self) -> int:
        return len(self.image_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample_id, degradation_suffix = self.image_index[idx]

        # Load image
        image = self._load_image(sample_id, degradation_suffix)

        # Load signals
        signals, lead_mask = self._load_signals(sample_id)

        # Apply transform to image only (signals don't need spatial transforms)
        if self.transform:
            # Transform expects (image, mask) but we only have image
            # Create a dummy mask for transform compatibility
            dummy_mask = torch.zeros(3, image.shape[1], image.shape[2])
            image, _ = self.transform(image, dummy_mask)

        # Metadata
        metadata = {
            "sample_id": sample_id,
            "degradation_type": degradation_suffix,
            "fs": self.target_fs,
        }

        return image, signals, lead_mask, metadata


class KaggleECGDatasetSimple(torch.utils.data.Dataset[Any]):
    """
    Simplified dataset that returns only (image, signals, lead_mask).

    Compatible with standard PyTorch training loops that expect
    (input, target) pairs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._dataset = KaggleECGDataset(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, signals, lead_mask, _ = self._dataset[idx]
        return image, signals, lead_mask
