"""
Physiological constraint enforcement for ECG signals.

This module implements known physiological relationships between ECG leads
to improve signal quality and consistency.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class PhysiologicalConstraints:
    """Enforce physiological constraints on 12-lead ECG signals."""

    # Standard 12-lead order
    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Lead indices
    I_IDX = 0
    II_IDX = 1
    III_IDX = 2
    AVR_IDX = 3
    AVL_IDX = 4
    AVF_IDX = 5

    def __init__(self, alpha: float = 0.3):
        """Initialize constraint enforcer.

        Args:
            alpha: Weight for constraint enforcement (0 = no enforcement, 1 = full enforcement)
        """
        self.alpha = alpha

    def apply_goldberger_equations(self, signals: torch.Tensor) -> torch.Tensor:
        """Apply Goldberger's equations to enforce limb lead relationships.

        Relationships:
        - III = II - I
        - aVR = -(I + II) / 2
        - aVL = I - II / 2
        - aVF = II - I / 2

        Args:
            signals: Tensor of shape (12, n_samples) with ECG signals

        Returns:
            Corrected signals with enforced relationships
        """
        if signals.shape[0] < 6:
            return signals  # Not enough leads to apply constraints

        signals = signals.clone()

        # Extract limb leads
        I = signals[self.I_IDX]
        II = signals[self.II_IDX]
        III = signals[self.III_IDX]
        aVR = signals[self.AVR_IDX]
        aVL = signals[self.AVL_IDX]
        aVF = signals[self.AVF_IDX]

        # Compute expected values from relationships
        III_expected = II - I
        aVR_expected = -(I + II) / 2
        aVL_expected = I - II / 2
        aVF_expected = II - I / 2

        # Blend with original values
        signals[self.III_IDX] = (1 - self.alpha) * III + self.alpha * III_expected
        signals[self.AVR_IDX] = (1 - self.alpha) * aVR + self.alpha * aVR_expected
        signals[self.AVL_IDX] = (1 - self.alpha) * aVL + self.alpha * aVL_expected
        signals[self.AVF_IDX] = (1 - self.alpha) * aVF + self.alpha * aVF_expected

        return signals

    def validate_heart_rate_consistency(
        self, signals: torch.Tensor, fs: float, tolerance: float = 30.0
    ) -> dict[str, float]:
        """Validate that heart rate is consistent across leads.

        Args:
            signals: Tensor of shape (12, n_samples)
            fs: Sampling frequency in Hz
            tolerance: Maximum allowed heart rate difference in BPM (default: 30.0 for robustness)

        Returns:
            Dictionary with validation results
        """
        from scipy.signal import find_peaks

        heart_rates = []

        for i in range(signals.shape[0]):
            signal = signals[i].cpu().numpy()

            # Skip if signal is mostly NaN
            if np.isnan(signal).sum() > len(signal) * 0.5:
                continue

            # Remove NaN values
            valid_mask = ~np.isnan(signal)
            valid_signal = signal[valid_mask]

            if len(valid_signal) < 100:
                continue

            # Normalize signal
            valid_signal = (valid_signal - np.mean(valid_signal)) / (np.std(valid_signal) + 1e-6)

            # Find peaks (QRS complexes)
            # Typical QRS is the highest amplitude feature
            peaks, _ = find_peaks(valid_signal, distance=int(0.5 * fs), prominence=0.5)

            if len(peaks) > 1:
                # Calculate heart rate
                rr_intervals = np.diff(peaks) / fs  # in seconds
                mean_rr = np.mean(rr_intervals)
                hr = 60 / mean_rr
                heart_rates.append(hr)

        if len(heart_rates) == 0:
            return {"valid": False, "reason": "No heart rate detected"}

        mean_hr = np.mean(heart_rates)
        std_hr = np.std(heart_rates)
        max_diff = np.max(np.abs(np.array(heart_rates) - mean_hr))

        return {
            "valid": max_diff < tolerance,
            "mean_hr": mean_hr,
            "std_hr": std_hr,
            "max_diff": max_diff,
            "heart_rates": heart_rates,
        }

    def check_amplitude_plausibility(self, signals: torch.Tensor) -> dict[str, bool]:
        """Check if signal amplitudes are physiologically plausible.

        Typical ECG amplitudes (in mV):
        - P wave: 0.05-0.25 mV
        - QRS complex: 0.5-2.5 mV
        - T wave: 0.1-0.5 mV

        Args:
            signals: Tensor of shape (12, n_samples) in mV

        Returns:
            Dictionary with plausibility checks
        """
        # Remove NaN values for amplitude calculation
        valid_signals = []
        for i in range(signals.shape[0]):
            signal = signals[i]
            valid_mask = ~torch.isnan(signal)
            if valid_mask.sum() > 0:
                valid_signals.append(signal[valid_mask])

        if len(valid_signals) == 0:
            return {"valid": False, "reason": "All signals are NaN"}

        # Concatenate all valid samples
        all_values = torch.cat(valid_signals)

        # Calculate peak-to-peak amplitude
        p2p_amplitude = (all_values.max() - all_values.min()).item()

        # Check if within plausible range (0.05 mV to 6 mV for full ECG)
        # Some patients naturally have larger QRS amplitudes (e.g., in chest leads)
        plausible_min = 0.05  # mV
        plausible_max = 6.0  # mV

        return {
            "valid": plausible_min < p2p_amplitude < plausible_max,
            "p2p_amplitude": p2p_amplitude,
            "min_expected": plausible_min,
            "max_expected": plausible_max,
        }

    def optimize_cross_lead_consistency(
        self, signals: torch.Tensor, iterations: int = 5
    ) -> torch.Tensor:
        """Iteratively optimize signals to satisfy physiological constraints.

        Args:
            signals: Tensor of shape (12, n_samples)
            iterations: Number of optimization iterations

        Returns:
            Optimized signals
        """
        signals = signals.clone()

        for _ in range(iterations):
            # Apply Goldberger equations
            signals = self.apply_goldberger_equations(signals)

            # Additional constraint: Precordial leads should show R-wave progression
            # (V1 typically negative, V6 typically positive)
            # This is a soft constraint - we just encourage the trend
            if signals.shape[0] >= 12:
                # Get precordial leads V1-V6 (indices 6-11)
                v_leads = signals[6:12]

                # Calculate mean amplitude for each V lead
                v_means = torch.nanmean(v_leads, dim=1)

                # Expect increasing trend from V1 to V6
                # If reversed, might be mirrored - but we don't force it too hard
                if not torch.isnan(v_means).any():
                    # Slight adjustment toward expected progression
                    expected_progression = torch.linspace(
                        v_means.min(), v_means.max(), 6, device=signals.device
                    )
                    adjustment = (expected_progression - v_means) * 0.05  # Very soft constraint
                    signals[6:12] = signals[6:12] + adjustment.unsqueeze(1)

        return signals


class PhysiologicalLoss(nn.Module):
    """Loss function that penalizes physiologically implausible signals."""

    def __init__(self, weight: float = 0.1):
        """Initialize loss.

        Args:
            weight: Weight for physiological loss term
        """
        super().__init__()
        self.weight = weight
        self.constraints = PhysiologicalConstraints(alpha=0.0)  # Just for validation

    def forward(self, pred_signals: torch.Tensor, target_signals: torch.Tensor) -> torch.Tensor:
        """Compute physiological constraint violation loss.

        Args:
            pred_signals: Predicted signals (batch, 12, n_samples)
            target_signals: Target signals (batch, 12, n_samples)

        Returns:
            Loss value
        """
        batch_size = pred_signals.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            pred = pred_signals[i]
            target = target_signals[i]

            # Extract limb leads
            I_pred = pred[0]
            II_pred = pred[1]
            III_pred = pred[2]
            aVR_pred = pred[3]
            aVL_pred = pred[4]
            aVF_pred = pred[5]

            # Compute expected values
            III_expected = II_pred - I_pred
            aVR_expected = -(I_pred + II_pred) / 2
            aVL_expected = I_pred - II_pred / 2
            aVF_expected = II_pred - I_pred / 2

            # Compute violations
            violation_III = torch.mean((III_pred - III_expected) ** 2)
            violation_aVR = torch.mean((aVR_pred - aVR_expected) ** 2)
            violation_aVL = torch.mean((aVL_pred - aVL_expected) ** 2)
            violation_aVF = torch.mean((aVF_pred - aVF_expected) ** 2)

            total_loss += violation_III + violation_aVR + violation_aVL + violation_aVF

        return self.weight * total_loss / batch_size


def apply_constraints_to_predictions(
    signals: np.ndarray, fs: float = 500.0, alpha: float = 0.3
) -> np.ndarray:
    """Apply physiological constraints to ECG predictions.

    This is a convenience function for post-processing predictions.

    Args:
        signals: NumPy array of shape (12, n_samples) in mV
        fs: Sampling frequency in Hz
        alpha: Constraint strength (0 = no change, 1 = full enforcement)

    Returns:
        Corrected signals as NumPy array
    """
    # Convert to torch
    signals_torch = torch.from_numpy(signals).float()

    # Apply constraints
    constraints = PhysiologicalConstraints(alpha=alpha)
    corrected = constraints.apply_goldberger_equations(signals_torch)
    corrected = constraints.optimize_cross_lead_consistency(corrected, iterations=3)

    # Validate
    hr_validation = constraints.validate_heart_rate_consistency(corrected, fs)
    amp_validation = constraints.check_amplitude_plausibility(corrected)

    print(f"Heart rate validation: {hr_validation}")
    print(f"Amplitude validation: {amp_validation}")

    return corrected.numpy()


if __name__ == "__main__":
    # Example usage
    print("Testing physiological constraints...")

    # Create synthetic 12-lead ECG (random for demonstration)
    n_samples = 5000
    signals = torch.randn(12, n_samples) * 0.5  # Random signals in mV

    # Make them satisfy approximate relationships initially
    signals[2] = signals[1] - signals[0] + torch.randn(n_samples) * 0.1  # III ≈ II - I
    signals[3] = -(signals[0] + signals[1]) / 2 + torch.randn(n_samples) * 0.1  # aVR

    print(f"Input signals shape: {signals.shape}")

    # Apply constraints
    constraints = PhysiologicalConstraints(alpha=0.5)
    corrected = constraints.apply_goldberger_equations(signals)
    corrected = constraints.optimize_cross_lead_consistency(corrected)

    print(f"Output signals shape: {corrected.shape}")

    # Validate
    hr_check = constraints.validate_heart_rate_consistency(corrected, fs=500.0)
    amp_check = constraints.check_amplitude_plausibility(corrected)

    print(f"\nHeart rate validation: {hr_check}")
    print(f"Amplitude validation: {amp_check}")

    print("\n✅ Physiological constraints module ready!")
