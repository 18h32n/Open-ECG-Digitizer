"""
Differentiable signal extraction head for end-to-end ECG training.

This module extracts 12-lead ECG signals from UNet segmentation output
using differentiable operations, enabling gradient-based training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# Standard 12-lead ECG layout (3 rows of 4 leads + 1 rhythm strip)
# Row 1: I, aVR, V1, V4
# Row 2: II, aVL, V2, V5
# Row 3: III, aVF, V3, V6
# Row 4: II (rhythm strip - full 10s)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Standard ECG layout mapping
# Each tuple is (row_index, column_index, lead_index)
# Row indices: 0-2 for 3x4 grid, 3 for rhythm strip
STANDARD_LAYOUT = {
    "I": (0, 0, 0),
    "aVR": (0, 1, 3),
    "V1": (0, 2, 6),
    "V4": (0, 3, 9),
    "II": (1, 0, 1),
    "aVL": (1, 1, 4),
    "V2": (1, 2, 7),
    "V5": (1, 3, 10),
    "III": (2, 0, 2),
    "aVF": (2, 1, 5),
    "V3": (2, 2, 8),
    "V6": (2, 3, 11),
}


class SoftArgmax1D(nn.Module):
    """Differentiable soft argmax along a single dimension."""

    def __init__(self, beta: float = 10.0):
        """
        Args:
            beta: Temperature parameter. Higher = sharper (closer to hard argmax).
        """
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute soft argmax along specified dimension.

        Args:
            x: Input tensor
            dim: Dimension to compute argmax over

        Returns:
            Soft argmax indices (same shape as input but dimension `dim` reduced)
        """
        # Create index tensor
        n = x.shape[dim]
        indices = torch.arange(n, dtype=x.dtype, device=x.device)

        # Reshape indices for broadcasting
        shape = [1] * x.ndim
        shape[dim] = n
        indices = indices.view(shape)

        # Compute softmax weights
        weights = F.softmax(self.beta * x, dim=dim)

        # Weighted sum of indices
        return (weights * indices).sum(dim=dim)


class DifferentiableSignalHead(nn.Module):
    """
    Differentiable signal extraction from UNet segmentation output.

    Extracts 12-lead ECG signals using soft argmax and learned layout mapping.
    """

    def __init__(
        self,
        signal_length: int = 5000,
        num_leads: int = 12,
        signal_class: int = 2,
        num_rows: int = 4,
        num_cols: int = 4,
        soft_argmax_beta: float = 20.0,
        use_learned_layout: bool = False,
    ):
        """
        Initialize the signal head.

        Args:
            signal_length: Target output signal length per lead
            num_leads: Number of ECG leads (default 12)
            signal_class: Index of signal class in segmentation output
            num_rows: Number of rows in ECG layout
            num_cols: Number of columns in ECG layout (for 3x4 grid)
            soft_argmax_beta: Temperature for soft argmax
            use_learned_layout: If True, learn layout mapping; else use fixed layout
        """
        super().__init__()
        self.signal_length = signal_length
        self.num_leads = num_leads
        self.signal_class = signal_class
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.soft_argmax = SoftArgmax1D(beta=soft_argmax_beta)
        self.use_learned_layout = use_learned_layout

        # Learnable vertical position refinement per row
        self.row_offsets = nn.Parameter(torch.zeros(num_rows))

        # Scale factor for converting pixel positions to mV
        # Will be learned or set based on calibration
        self.mv_per_pixel = nn.Parameter(torch.tensor(0.01))  # Initialize around typical scale

        # Optional learned layout (which image region maps to which lead)
        if use_learned_layout:
            # (num_rows, num_cols) -> lead_idx mapping, learnable
            self.layout_logits = nn.Parameter(torch.randn(num_rows, num_cols, num_leads))

    def forward(
        self,
        seg_output: torch.Tensor,
        image_height: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract signals from segmentation output.

        Args:
            seg_output: UNet output (batch, num_classes, H, W)
            image_height: Original image height for scale computation

        Returns:
            signals: Extracted signals (batch, 12, signal_length) in mV
            lead_mask: Valid sample mask (batch, 12, signal_length)
        """
        batch_size = seg_output.shape[0]
        device = seg_output.device

        # Get signal probability map
        seg_probs = F.softmax(seg_output, dim=1)
        signal_prob = seg_probs[:, self.signal_class]  # (batch, H, W)

        # Extract signals using soft argmax per column
        # signal_prob shape: (batch, H, W)
        # We want y-position per column: (batch, W)
        y_positions = self.soft_argmax(signal_prob, dim=1)  # (batch, W)

        # Partition into rows and columns based on standard ECG layout
        signals, lead_mask = self._partition_and_extract(
            y_positions, signal_prob, batch_size, device
        )

        return signals, lead_mask

    def _partition_and_extract(
        self,
        y_positions: torch.Tensor,
        signal_prob: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Partition y-positions into ECG leads based on layout.

        Args:
            y_positions: (batch, W) pixel y-coordinates of signal
            signal_prob: (batch, H, W) probability map for confidence weighting
            batch_size: Number of samples in batch
            device: Torch device

        Returns:
            signals: (batch, 12, signal_length)
            lead_mask: (batch, 12, signal_length)
        """
        H, W = signal_prob.shape[1], signal_prob.shape[2]

        # Initialize output tensors
        signals = torch.zeros(batch_size, self.num_leads, self.signal_length, device=device)
        lead_mask = torch.zeros(batch_size, self.num_leads, self.signal_length, device=device)

        # Estimate row boundaries (divide image into 4 rows: 3 short leads + 1 rhythm strip)
        # Standard layout: rows 0-2 are equal height, row 3 (rhythm strip) may be smaller
        row_height = H // 4
        col_width = W // 4

        # Process each lead region
        for lead_name, (row_idx, col_idx, lead_idx) in STANDARD_LAYOUT.items():
            # Determine signal length for this lead
            if lead_idx == 1:  # Lead II
                # Lead II appears in row 1 (2.5s) AND row 3 (rhythm strip, full 10s)
                # For training, we'll use the rhythm strip (row 3) for full duration
                # The 2.5s segment from row 1 is handled below
                pass

            # Get row boundaries
            row_start = row_idx * row_height
            row_end = (row_idx + 1) * row_height

            # Get column boundaries (each column is 2.5s of signal)
            col_start = col_idx * col_width
            col_end = (col_idx + 1) * col_width

            # Extract y-positions for this region
            region_y = y_positions[:, col_start:col_end]  # (batch, region_width)

            # Get signal probability for confidence weighting
            region_prob = signal_prob[:, row_start:row_end, col_start:col_end].mean(dim=1)

            # Convert y-positions to baseline-centered values
            # Subtract row center to get relative position
            row_center = (row_start + row_end) / 2 + self.row_offsets[row_idx]
            relative_y = region_y - row_center

            # Scale to mV (assuming typical calibration: 10mm/mV, need to estimate pixel scale)
            # Positive y in image is downward, but ECG positive is upward, so negate
            signal_values = -relative_y * self.mv_per_pixel

            # Resample to target length
            # For 2.5s leads, target_length = signal_length // 4
            if lead_name == "II" and row_idx == 1:
                # Short II segment (2.5s from row 1)
                target_len = self.signal_length // 4
                resampled = self._resample_signal(signal_values, target_len)
                signals[:, lead_idx, :target_len] = resampled
                lead_mask[:, lead_idx, :target_len] = 1.0
            else:
                # Standard 2.5s lead
                target_len = self.signal_length // 4
                resampled = self._resample_signal(signal_values, target_len)
                signals[:, lead_idx, :target_len] = resampled
                lead_mask[:, lead_idx, :target_len] = 1.0

        # Handle Lead II rhythm strip (row 3, full width, full 10s)
        row_start = 3 * row_height
        row_end = H
        rhythm_y = y_positions  # Full width for rhythm strip

        row_center = (row_start + row_end) / 2 + self.row_offsets[3]

        # The rhythm strip takes up the full width - extract from row 3 region
        # Get y-positions weighted by signal probability in rhythm strip region
        rhythm_prob = signal_prob[:, row_start:row_end, :]  # (batch, row_height, W)
        rhythm_weights = F.softmax(rhythm_prob.sum(dim=1) * 10, dim=1)  # (batch, W)

        # Weighted average y-position per column in rhythm strip region
        y_range = torch.arange(row_start, row_end, device=device, dtype=signal_prob.dtype)
        y_range = y_range.view(1, -1, 1).expand(batch_size, -1, W)
        rhythm_y_weighted = (rhythm_prob * y_range).sum(dim=1) / (rhythm_prob.sum(dim=1) + 1e-6)

        # Convert to mV
        rhythm_signal = -(rhythm_y_weighted - row_center) * self.mv_per_pixel

        # Resample to full signal length
        rhythm_resampled = self._resample_signal(rhythm_signal, self.signal_length)

        # Overwrite Lead II with rhythm strip data (full 10s)
        signals[:, 1, :] = rhythm_resampled
        lead_mask[:, 1, :] = 1.0

        return signals, lead_mask

    def _resample_signal(
        self, signal: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """
        Resample signal to target length using interpolation.

        Args:
            signal: (batch, length) input signal
            target_length: Target output length

        Returns:
            Resampled signal (batch, target_length)
        """
        # Add channel dimension for interpolate
        signal = signal.unsqueeze(1)  # (batch, 1, length)

        # Use bilinear interpolation (linear for 1D)
        resampled = F.interpolate(
            signal, size=target_length, mode="linear", align_corners=True
        )

        return resampled.squeeze(1)  # (batch, target_length)


class SimpleDifferentiableSignalHead(nn.Module):
    """
    Simplified differentiable signal head for quick experimentation.

    Uses a straightforward soft argmax approach without complex layout parsing.
    Suitable for initial training and debugging.
    """

    def __init__(
        self,
        signal_length: int = 5000,
        num_leads: int = 12,
        signal_class: int = 2,
        soft_argmax_beta: float = 20.0,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.num_leads = num_leads
        self.signal_class = signal_class
        self.soft_argmax = SoftArgmax1D(beta=soft_argmax_beta)

        # Learnable parameters for extracting 12 leads from image
        # Maps image columns to lead outputs
        self.lead_projector = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(signal_length),
            nn.Conv1d(64, num_leads, kernel_size=1),
        )

        # Scale factor
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(
        self, seg_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract signals from segmentation output.

        Args:
            seg_output: (batch, num_classes, H, W) UNet output

        Returns:
            signals: (batch, 12, signal_length) in mV
            lead_mask: (batch, 12, signal_length) validity mask
        """
        batch_size = seg_output.shape[0]
        H, W = seg_output.shape[2], seg_output.shape[3]
        device = seg_output.device

        # Get signal probability
        seg_probs = F.softmax(seg_output, dim=1)
        signal_prob = seg_probs[:, self.signal_class]  # (batch, H, W)

        # Soft argmax to get y-position per column
        y_positions = self.soft_argmax(signal_prob, dim=1)  # (batch, W)

        # Center around image middle
        y_centered = y_positions - H / 2  # (batch, W)

        # Scale to approximate mV range
        y_scaled = y_centered * self.scale  # (batch, W)

        # Project to 12 leads
        y_expanded = y_scaled.unsqueeze(1)  # (batch, 1, W)
        signals = self.lead_projector(y_expanded)  # (batch, 12, signal_length)

        # Create mask (all valid for simplified version)
        # Lead II gets full length, others get quarter length
        lead_mask = torch.zeros(batch_size, self.num_leads, self.signal_length, device=device)
        lead_mask[:, 1, :] = 1.0  # Lead II full length
        lead_mask[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :self.signal_length // 4] = 1.0

        return signals, lead_mask


class UNetWithSignalHead(nn.Module):
    """
    Combined UNet + Signal Head for end-to-end training.

    Wraps the UNet segmentation model with a differentiable signal head
    for direct image-to-signal training.
    """

    def __init__(
        self,
        unet: nn.Module,
        signal_head: Optional[nn.Module] = None,
        signal_length: int = 5000,
        freeze_unet: bool = False,
    ):
        """
        Args:
            unet: Pre-trained or new UNet model
            signal_head: Signal extraction head (created if None)
            signal_length: Target signal length
            freeze_unet: Whether to freeze UNet weights
        """
        super().__init__()
        self.unet = unet

        if signal_head is None:
            signal_head = SimpleDifferentiableSignalHead(signal_length=signal_length)
        self.signal_head = signal_head

        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass from image to signals.

        Args:
            image: (batch, 3, H, W) RGB image

        Returns:
            seg_output: (batch, 4, H, W) segmentation logits
            signals: (batch, 12, signal_length) extracted signals
            lead_mask: (batch, 12, signal_length) validity mask
        """
        # Get segmentation
        seg_output = self.unet(image)

        # Extract signals
        signals, lead_mask = self.signal_head(seg_output)

        return seg_output, signals, lead_mask
