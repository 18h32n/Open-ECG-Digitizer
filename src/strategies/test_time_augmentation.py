"""
Test-Time Augmentation (TTA) for ECG digitization.

Applies multiple augmentations at inference time and averages predictions
to improve robustness and accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, List, Tuple
from scipy.signal import resample


class TestTimeAugmentation:
    """Test-time augmentation wrapper for ECG digitization models."""

    def __init__(
        self,
        model: Any,
        n_augmentations: int = 8,
        flip_horizontal: bool = True,
        flip_vertical: bool = False,
        rotations: List[float] = [-2, -1, 0, 1, 2],  # degrees
        scales: List[float] = [0.95, 1.0, 1.05],
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
    ):
        """Initialize TTA.

        Args:
            model: Inference wrapper model
            n_augmentations: Number of augmented predictions to generate
            flip_horizontal: Whether to include horizontal flips
            flip_vertical: Whether to include vertical flips
            rotations: List of rotation angles in degrees
            scales: List of scale factors
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
        """
        self.model = model
        self.n_augmentations = n_augmentations
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.rotations = rotations
        self.scales = scales
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        # Generate augmentation configurations
        self.aug_configs = self._generate_augmentation_configs()

    def _generate_augmentation_configs(self) -> List[dict]:
        """Generate diverse augmentation configurations.

        Returns:
            List of augmentation config dictionaries
        """
        configs = []

        # Always include identity (no augmentation)
        configs.append({
            "flip_h": False,
            "flip_v": False,
            "rotation": 0,
            "scale": 1.0,
            "brightness": 1.0,
            "contrast": 1.0,
        })

        # Generate random configurations
        np.random.seed(42)  # For reproducibility
        for _ in range(self.n_augmentations - 1):
            config = {
                "flip_h": np.random.choice([True, False]) if self.flip_horizontal else False,
                "flip_v": np.random.choice([True, False]) if self.flip_vertical else False,
                "rotation": np.random.choice(self.rotations),
                "scale": np.random.choice(self.scales),
                "brightness": np.random.uniform(*self.brightness_range),
                "contrast": np.random.uniform(*self.contrast_range),
            }
            configs.append(config)

        return configs

    def _apply_augmentation(self, image: torch.Tensor, config: dict) -> torch.Tensor:
        """Apply augmentation to image.

        Args:
            image: Input image tensor (1, 3, H, W)
            config: Augmentation configuration

        Returns:
            Augmented image
        """
        aug_image = image.clone()

        # Flip horizontal
        if config["flip_h"]:
            aug_image = torch.flip(aug_image, dims=[3])

        # Flip vertical
        if config["flip_v"]:
            aug_image = torch.flip(aug_image, dims=[2])

        # Rotation
        if config["rotation"] != 0:
            angle = config["rotation"]
            # Convert angle to radians
            theta = torch.tensor([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
            ], dtype=torch.float32).unsqueeze(0)

            grid = F.affine_grid(theta, aug_image.size(), align_corners=False)
            aug_image = F.grid_sample(aug_image, grid, align_corners=False)

        # Scale
        if config["scale"] != 1.0:
            h, w = aug_image.shape[2:]
            new_h = int(h * config["scale"])
            new_w = int(w * config["scale"])
            aug_image = F.interpolate(aug_image, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # Pad or crop to original size
            if config["scale"] < 1.0:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                aug_image = F.pad(aug_image, (pad_w, pad_w, pad_h, pad_h), value=0)
            else:
                # Crop
                crop_h = (new_h - h) // 2
                crop_w = (new_w - w) // 2
                aug_image = aug_image[:, :, crop_h:crop_h + h, crop_w:crop_w + w]

        # Brightness
        if config["brightness"] != 1.0:
            aug_image = aug_image * config["brightness"]
            aug_image = torch.clamp(aug_image, 0, 1)

        # Contrast
        if config["contrast"] != 1.0:
            mean = aug_image.mean()
            aug_image = (aug_image - mean) * config["contrast"] + mean
            aug_image = torch.clamp(aug_image, 0, 1)

        return aug_image

    def _reverse_signal_augmentation(
        self, signal: torch.Tensor, config: dict, original_length: int
    ) -> torch.Tensor:
        """Reverse augmentations applied to image for signal space.

        Args:
            signal: Predicted signal tensor (12, n_samples)
            config: Augmentation configuration used
            original_length: Original signal length before resampling

        Returns:
            De-augmented signal
        """
        signal = signal.clone()

        # Reverse horizontal flip (reverses time)
        if config["flip_h"]:
            signal = torch.flip(signal, dims=[1])

        # Reverse vertical flip (inverts amplitude)
        if config["flip_v"]:
            signal = -signal

        # Note: Rotation and scale don't directly affect 1D signal in our pipeline
        # The model internally handles these transformations

        return signal

    def __call__(
        self, image: torch.Tensor, layout_should_include_substring: str | None = None
    ) -> dict[str, Any]:
        """Run inference with test-time augmentation.

        Args:
            image: Input image tensor (1, 3, H, W)
            layout_should_include_substring: Optional layout filter

        Returns:
            Dictionary with averaged predictions and metadata
        """
        all_predictions = []
        all_confidences = []

        print(f"Running TTA with {len(self.aug_configs)} augmentations...")

        for i, config in enumerate(self.aug_configs):
            # Apply augmentation
            aug_image = self._apply_augmentation(image, config)

            # Run model
            result = self.model(aug_image, layout_should_include_substring=layout_should_include_substring)

            # Extract canonical signals
            canonical = result.get("signal", {}).get("canonical_lines")
            if canonical is None:
                canonical = result.get("canonical_lines")

            if canonical is not None:
                # Reverse augmentations for signal
                original_length = canonical.shape[1] if canonical.dim() > 1 else len(canonical)
                canonical = self._reverse_signal_augmentation(canonical, config, original_length)

                all_predictions.append(canonical)

                # Use matching cost as confidence (lower is better)
                cost = result.get("signal", {}).get("layout_matching_cost", 1.0)
                confidence = 1.0 / (1.0 + cost)  # Convert cost to confidence
                all_confidences.append(confidence)

            print(f"  Aug {i+1}/{len(self.aug_configs)}: cost={cost:.3f}, confidence={confidence:.3f}")

        if len(all_predictions) == 0:
            print("Warning: No valid predictions from TTA")
            return result  # Return last result

        # Stack predictions
        predictions_tensor = torch.stack(all_predictions)  # (n_augs, 12, n_samples)
        confidences_tensor = torch.tensor(all_confidences).unsqueeze(-1).unsqueeze(-1)  # (n_augs, 1, 1)

        # Normalize confidences
        confidences_tensor = confidences_tensor / confidences_tensor.sum()

        # Weighted average
        averaged_prediction = (predictions_tensor * confidences_tensor).sum(dim=0)

        # Update result with averaged prediction
        result["signal"]["canonical_lines"] = averaged_prediction
        result["canonical_lines"] = averaged_prediction
        result["tta_predictions"] = predictions_tensor
        result["tta_confidences"] = confidences_tensor.squeeze()
        result["tta_n_augmentations"] = len(all_predictions)

        print(f"✅ TTA complete: {len(all_predictions)} predictions averaged")

        return result


class EnsembleTTA:
    """Ensemble multiple models with TTA."""

    def __init__(
        self,
        models: List[Any],
        model_weights: List[float] | None = None,
        tta_per_model: int = 5,
    ):
        """Initialize ensemble with TTA.

        Args:
            models: List of inference wrapper models
            model_weights: Optional weights for each model (will normalize)
            tta_per_model: Number of TTA augmentations per model
        """
        self.models = models
        self.tta_wrappers = [
            TestTimeAugmentation(model, n_augmentations=tta_per_model)
            for model in models
        ]

        if model_weights is None:
            model_weights = [1.0] * len(models)

        # Normalize weights
        total_weight = sum(model_weights)
        self.model_weights = [w / total_weight for w in model_weights]

    def __call__(
        self, image: torch.Tensor, layout_should_include_substring: str | None = None
    ) -> dict[str, Any]:
        """Run ensemble inference with TTA.

        Args:
            image: Input image tensor
            layout_should_include_substring: Optional layout filter

        Returns:
            Ensemble prediction
        """
        all_model_predictions = []
        all_model_confidences = []

        for i, (tta_wrapper, weight) in enumerate(zip(self.tta_wrappers, self.model_weights)):
            print(f"\nModel {i+1}/{len(self.models)} (weight={weight:.3f}):")
            result = tta_wrapper(image, layout_should_include_substring)

            canonical = result.get("signal", {}).get("canonical_lines")
            if canonical is not None:
                all_model_predictions.append(canonical)
                all_model_confidences.append(weight)

        if len(all_model_predictions) == 0:
            print("Warning: No valid predictions from ensemble")
            return result

        # Stack and weight
        predictions_tensor = torch.stack(all_model_predictions)
        weights_tensor = torch.tensor(all_model_confidences).unsqueeze(-1).unsqueeze(-1)

        # Weighted average
        final_prediction = (predictions_tensor * weights_tensor).sum(dim=0)

        # Update result
        result["signal"]["canonical_lines"] = final_prediction
        result["canonical_lines"] = final_prediction
        result["ensemble_n_models"] = len(all_model_predictions)

        return result


def create_tta_inference_wrapper(config: Any, n_augmentations: int = 10) -> TestTimeAugmentation:
    """Create TTA-enabled inference wrapper from config.

    Args:
        config: Model configuration
        n_augmentations: Number of augmentations

    Returns:
        TTA wrapper
    """
    from src.utils import import_class_from_path

    # Create base model
    inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
    base_model = inference_wrapper_class(**config.MODEL.KWARGS)

    # Wrap with TTA
    tta_model = TestTimeAugmentation(
        model=base_model,
        n_augmentations=n_augmentations,
        flip_horizontal=True,
        flip_vertical=False,  # Usually don't want to flip ECGs vertically
        rotations=[-2, -1, 0, 1, 2],
        scales=[0.97, 1.0, 1.03],
        brightness_range=(0.95, 1.05),
        contrast_range=(0.95, 1.05),
    )

    return tta_model


if __name__ == "__main__":
    print("Test-Time Augmentation module ready!")
    print("\nUsage example:")
    print("""
    from src.strategies.test_time_augmentation import create_tta_inference_wrapper
    from src.config.default import get_cfg
    from src.utils import find_config_path

    # Load config
    config_path = find_config_path('kaggle_inference.yml')
    cfg = get_cfg(config_path)

    # Create TTA-enabled model
    tta_model = create_tta_inference_wrapper(cfg, n_augmentations=10)

    # Use like normal model
    result = tta_model(image, layout_should_include_substring=None)
    """)
