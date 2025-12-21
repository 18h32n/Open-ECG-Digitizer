"""
Signal-Level Augmentation for Test-Time Augmentation (TTA).

This module provides augmentation functions that operate on extracted ECG signals
rather than raw images or canonical tensors. This allows TTA to be applied even
when the canonical tensor contains high NaN content from grid layout patterns.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import resample


def apply_time_warping(
    signal: npt.NDArray[Any],
    warp_factor: float = 0.02,
) -> npt.NDArray[Any]:
    """Apply time warping augmentation to a signal.

    Args:
        signal: Input signal array.
        warp_factor: Warping strength (±2% recommended). Positive = stretch, negative = compress.

    Returns:
        Time-warped signal with original length.
    """
    original_length = len(signal)
    warped_length = int(original_length * (1 + warp_factor))

    # Resample to warped length, then back to original
    warped = resample(signal, warped_length)
    return resample(warped, original_length)


def apply_amplitude_scaling(
    signal: npt.NDArray[Any],
    scale_factor: float = 0.05,
) -> npt.NDArray[Any]:
    """Apply amplitude scaling augmentation to a signal.

    Args:
        signal: Input signal array.
        scale_factor: Scaling strength (±5% recommended).

    Returns:
        Amplitude-scaled signal.
    """
    return signal * (1 + scale_factor)


def apply_baseline_shift(
    signal: npt.NDArray[Any],
    shift_value: float = 0.05,
) -> npt.NDArray[Any]:
    """Apply baseline shift augmentation to a signal.

    Args:
        signal: Input signal array.
        shift_value: Shift amount in mV (±0.05 mV recommended).

    Returns:
        Baseline-shifted signal.
    """
    return signal + shift_value


def apply_gaussian_noise(
    signal: npt.NDArray[Any],
    noise_std: float = 0.01,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """Apply Gaussian noise augmentation to a signal.

    Args:
        signal: Input signal array.
        noise_std: Standard deviation of noise (0.01 recommended).
        seed: Random seed for reproducibility (optional).

    Returns:
        Noisy signal.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_std, len(signal))
    return signal + noise


def apply_signal_augmentation(
    signals: dict[str, npt.NDArray[Any]],
    aug_idx: int,
    n_augmentations: int = 10,
) -> dict[str, npt.NDArray[Any]]:
    """Apply a specific augmentation to all signals.

    Args:
        signals: Dictionary of {lead_name: signal_array}.
        aug_idx: Augmentation index (0 to n_augmentations-1).
        n_augmentations: Total number of augmentations.

    Returns:
        Augmented signals dictionary.
    """
    # Map augmentation index to parameters
    # We'll distribute augmentations across 4 types with varying strengths

    aug_type = aug_idx % 4
    strength_level = aug_idx // 4

    # Strength multiplier based on level (0 = baseline, 1 = weak, 2 = strong)
    strength_mult = [0.0, 0.5, 1.0, 1.5][strength_level % 4]

    augmented = {}

    for lead_name, signal in signals.items():
        if aug_type == 0:
            # Time warping: ±2% * strength_mult
            warp = 0.02 * strength_mult * (1 if aug_idx % 2 == 0 else -1)
            augmented[lead_name] = apply_time_warping(signal, warp)

        elif aug_type == 1:
            # Amplitude scaling: ±5% * strength_mult
            scale = 0.05 * strength_mult * (1 if aug_idx % 2 == 0 else -1)
            augmented[lead_name] = apply_amplitude_scaling(signal, scale)

        elif aug_type == 2:
            # Baseline shift: ±0.05 mV * strength_mult
            shift = 0.05 * strength_mult * (1 if aug_idx % 2 == 0 else -1)
            augmented[lead_name] = apply_baseline_shift(signal, shift)

        elif aug_type == 3:
            # Gaussian noise: 0.01 * strength_mult
            noise_std = 0.01 * strength_mult
            augmented[lead_name] = apply_gaussian_noise(signal, noise_std, seed=aug_idx)

    return augmented


def create_augmentation_set(
    signals: dict[str, npt.NDArray[Any]],
    n_augmentations: int = 10,
) -> list[dict[str, npt.NDArray[Any]]]:
    """Create a set of augmented signal versions.

    Args:
        signals: Dictionary of {lead_name: signal_array}.
        n_augmentations: Number of augmented versions to create.

    Returns:
        List of augmented signal dictionaries (includes original as first element).
    """
    aug_results = []

    # First element is always the original (no augmentation)
    aug_results.append(signals.copy())

    # Create n_augmentations - 1 augmented versions
    for aug_idx in range(1, n_augmentations):
        augmented = apply_signal_augmentation(signals, aug_idx, n_augmentations)
        aug_results.append(augmented)

    return aug_results


def average_augmented_results(
    aug_results: list[dict[str, npt.NDArray[Any]]],
) -> dict[str, npt.NDArray[Any]]:
    """Average results across multiple augmentations.

    Args:
        aug_results: List of augmented signal dictionaries.

    Returns:
        Averaged signals dictionary.
    """
    if not aug_results:
        raise ValueError("aug_results cannot be empty")

    # Get lead names from first result
    lead_names = list(aug_results[0].keys())

    # Average each lead across all augmentations
    averaged = {}
    for lead_name in lead_names:
        # Stack all augmented versions of this lead
        stacked = np.stack([result[lead_name] for result in aug_results])

        # Average across augmentations (axis 0)
        averaged[lead_name] = np.mean(stacked, axis=0)

    return averaged
