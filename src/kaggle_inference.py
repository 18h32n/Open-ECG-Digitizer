"""
Kaggle PhysioNet ECG Digitization Competition Inference Script

This script processes ECG images for the Kaggle competition and generates submissions.
It adapts the Open-ECG-Digitizer model output to match the competition requirements:
- Lead II: 10 seconds duration (floor(fs * 10) samples)
- Other leads: 2.5 seconds duration (floor(fs * 2.5) samples)
- Output units: millivolts (mV)
- Format: Long format with id,value pairs
"""

import argparse
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.signal import resample
from torchvision.io import decode_image
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg
from src.utils import find_config_path, import_class_from_path

# Standard 12-lead ECG order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II_INDEX = 1  # Index of Lead II in the standard order


def decode_and_prepare_image(file_path: str) -> torch.Tensor:
    """Load and prepare an image for inference.

    Args:
        file_path: Path to the image file.

    Returns:
        Prepared image tensor with shape (1, 3, H, W).
    """
    image: torch.Tensor = decode_image(file_path, mode="RGB")
    C, H, W = image.shape
    if C == 1:
        image = image.expand(3, H, W)
    elif C == 4:
        image = image[:3]
    return image.unsqueeze(0)


def extract_canonical_signals(got_values: dict[str, Any]) -> torch.Tensor | None:
    """Extract canonical 12-lead signals from model output.

    Args:
        got_values: Dictionary of model outputs.

    Returns:
        Canonical signals tensor with shape (12, n_samples) or None.
    """
    canonical: torch.Tensor | None = None
    if "signal" in got_values and isinstance(got_values["signal"], dict):
        canonical = got_values["signal"].get("canonical_lines")
    elif "canonical_lines" in got_values:
        canonical = got_values["canonical_lines"]
    return canonical


def resample_signal(signal: npt.NDArray[Any], target_length: int) -> npt.NDArray[Any]:
    """Resample a signal to target length using scipy.signal.resample.

    Args:
        signal: Input signal array.
        target_length: Target number of samples.

    Returns:
        Resampled signal.
    """
    if len(signal) == target_length:
        return signal
    return resample(signal, target_length)


def convert_units_uv_to_mv(signal: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert signal from microvolts (µV) to millivolts (mV).

    Args:
        signal: Signal in microvolts.

    Returns:
        Signal in millivolts.
    """
    return signal / 1000.0


def process_single_image(
    image_path: str,
    image_id: str,
    fs: float,
    inference_wrapper: Any,
) -> dict[str, npt.NDArray[Any]]:
    """Process a single ECG image and return digitized signals.

    Args:
        image_path: Path to the ECG image.
        image_id: Image identifier from test.csv.
        fs: Sampling frequency in Hz.
        inference_wrapper: Initialized inference wrapper model.

    Returns:
        Dictionary mapping lead names to signal arrays in mV.
    """
    # Load and prepare image
    image = decode_and_prepare_image(image_path)

    # Run inference
    got_values = inference_wrapper(image, layout_should_include_substring=None)

    # Extract canonical signals (shape: 12, n_samples) in µV
    canonical = extract_canonical_signals(got_values)

    if canonical is None:
        print(f"Warning: No signals extracted for {image_id}")
        # Return NaN arrays with correct shape
        lead_ii_samples = int(np.floor(fs * 10))
        other_samples = int(np.floor(fs * 2.5))

        results = {}
        for i, lead_name in enumerate(LEAD_NAMES):
            if i == LEAD_II_INDEX:
                results[lead_name] = np.full(lead_ii_samples, np.nan)
            else:
                results[lead_name] = np.full(other_samples, np.nan)
        return results

    # Convert to numpy and handle shape
    signals = canonical.squeeze().cpu().numpy()
    if signals.ndim == 1:
        signals = signals[None, :]

    # Calculate target lengths for each lead
    lead_ii_samples = int(np.floor(fs * 10))  # 10 seconds for Lead II
    other_samples = int(np.floor(fs * 2.5))   # 2.5 seconds for other leads

    # Process each lead
    results = {}
    for i, lead_name in enumerate(LEAD_NAMES):
        if i >= signals.shape[0]:
            # Lead not present in output
            target_samples = lead_ii_samples if i == LEAD_II_INDEX else other_samples
            results[lead_name] = np.full(target_samples, np.nan)
            continue

        signal = signals[i, :]

        # Determine target length based on lead
        if i == LEAD_II_INDEX:
            target_samples = lead_ii_samples
        else:
            target_samples = other_samples

        # Resample to target length
        resampled_signal = resample_signal(signal, target_samples)

        # Convert from µV to mV
        signal_mv = convert_units_uv_to_mv(resampled_signal)

        results[lead_name] = signal_mv

    return results


def create_submission_dataframe(
    test_metadata: pd.DataFrame,
    all_predictions: dict[str, dict[str, npt.NDArray[Any]]],
) -> pd.DataFrame:
    """Create submission dataframe in the required format.

    Format: id,value where id = {base_id}_{row_id}_{lead}

    Args:
        test_metadata: Test set metadata from test.csv.
        all_predictions: Dictionary mapping image IDs to lead signals.

    Returns:
        Submission dataframe with columns [id, value].
    """
    rows = []

    for image_id in all_predictions.keys():
        signals = all_predictions[image_id]

        for lead_name, signal in signals.items():
            for row_id, value in enumerate(signal):
                submission_id = f"{image_id}_{row_id}_{lead_name}"
                rows.append({"id": submission_id, "value": value})

    return pd.DataFrame(rows)


def main(config: CN) -> None:
    """Main function for Kaggle inference.

    Args:
        config: Configuration node.
    """
    # Load test metadata
    test_csv_path = config.DATA.test_csv_path
    test_images_dir = config.DATA.test_images_dir
    submission_path = config.DATA.submission_path

    print(f"Loading test metadata from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Get unique image IDs
    unique_ids = test_df["id"].unique()
    print(f"Found {len(unique_ids)} unique images to process")

    # Initialize inference wrapper
    print("Initializing inference model...")
    inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
    inference_wrapper = inference_wrapper_class(**config.MODEL.KWARGS)
    print("Model initialized successfully")

    # Process each image
    all_predictions = {}

    for image_id in tqdm(unique_ids, desc="Processing images"):
        # Get metadata for this image
        image_row = test_df[test_df["id"] == image_id].iloc[0]
        fs = image_row["fs"]

        # Construct image path
        image_path = os.path.join(test_images_dir, f"{image_id}.png")

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            # Create NaN predictions
            lead_ii_samples = int(np.floor(fs * 10))
            other_samples = int(np.floor(fs * 2.5))

            predictions = {}
            for i, lead_name in enumerate(LEAD_NAMES):
                if i == LEAD_II_INDEX:
                    predictions[lead_name] = np.full(lead_ii_samples, np.nan)
                else:
                    predictions[lead_name] = np.full(other_samples, np.nan)
            all_predictions[image_id] = predictions
            continue

        try:
            # Process the image
            predictions = process_single_image(
                image_path=image_path,
                image_id=image_id,
                fs=fs,
                inference_wrapper=inference_wrapper,
            )
            all_predictions[image_id] = predictions

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            # Create NaN predictions
            lead_ii_samples = int(np.floor(fs * 10))
            other_samples = int(np.floor(fs * 2.5))

            predictions = {}
            for i, lead_name in enumerate(LEAD_NAMES):
                if i == LEAD_II_INDEX:
                    predictions[lead_name] = np.full(lead_ii_samples, np.nan)
                else:
                    predictions[lead_name] = np.full(other_samples, np.nan)
            all_predictions[image_id] = predictions
            continue

    # Create submission dataframe
    print("Creating submission file...")
    submission_df = create_submission_dataframe(test_df, all_predictions)

    # Save submission
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(submission_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission for PhysioNet ECG Digitization Competition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="kaggle_inference.yml",
        help="Config file name or path (searched in . and src/config/)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config options like A.B.C=123",
    )
    args = parser.parse_args()

    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)

    if args.overrides:
        kv_list: list[str] = []
        for ov in args.overrides:
            if "=" not in ov:
                raise ValueError(f"Malformed override '{ov}'. Use KEY=VALUE")
            k, v = ov.split("=", 1)
            kv_list.extend([k, v])

        if kv_list:
            cfg.merge_from_list(kv_list)

    main(cfg)
