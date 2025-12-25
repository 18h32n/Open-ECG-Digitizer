"""
Signal reconstruction loss functions for ECG digitization training.

These losses are designed to supervise end-to-end training from images
to time-series signals using ground truth CSV data.
"""

import torch
import torch.nn as nn
from typing import Optional


class SignalReconstructionLoss(nn.Module):
    """
    Combined loss for ECG signal reconstruction.

    Components:
    - MSE loss for amplitude accuracy
    - Pearson correlation loss for waveform shape similarity
    - Goldberger equation loss for physiological constraints
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        correlation_weight: float = 0.5,
        goldberger_weight: float = 0.1,
        smooth: float = 1e-6,
    ):
        """
        Initialize the loss function.

        Args:
            mse_weight: Weight for MSE loss component
            correlation_weight: Weight for correlation loss component
            goldberger_weight: Weight for Goldberger equation loss
            smooth: Small value to avoid division by zero
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.goldberger_weight = goldberger_weight
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined signal reconstruction loss.

        Args:
            pred: Predicted signals (batch, 12, signal_length) in mV
            target: Target signals (batch, 12, signal_length) in mV
            mask: Optional binary mask (batch, 12, signal_length) for valid samples

        Returns:
            Combined loss scalar
        """
        if mask is None:
            mask = torch.ones_like(target)

        # MSE loss (masked)
        mse_loss = self._masked_mse(pred, target, mask)

        # Correlation loss (masked)
        corr_loss = self._masked_correlation_loss(pred, target, mask)

        # Goldberger equation loss
        goldberger_loss = self._goldberger_loss(pred, mask)

        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss
            + self.correlation_weight * corr_loss
            + self.goldberger_weight * goldberger_loss
        )

        return total_loss

    def _masked_mse(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss only on valid (masked) samples."""
        squared_diff = (pred - target) ** 2
        masked_squared_diff = squared_diff * mask
        valid_count = mask.sum() + self.smooth
        return masked_squared_diff.sum() / valid_count

    def _masked_correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 1 - Pearson correlation as loss, averaged over leads.

        This encourages correct waveform shape regardless of amplitude offset.
        """
        batch_size, n_leads, signal_length = pred.shape
        correlations = []

        for b in range(batch_size):
            for lead in range(n_leads):
                lead_mask = mask[b, lead]
                valid_count = lead_mask.sum()

                if valid_count < 10:  # Need enough samples for correlation
                    continue

                pred_lead = pred[b, lead] * lead_mask
                target_lead = target[b, lead] * lead_mask

                # Compute mean over valid samples
                pred_mean = pred_lead.sum() / (valid_count + self.smooth)
                target_mean = target_lead.sum() / (valid_count + self.smooth)

                # Center the signals
                pred_centered = (pred_lead - pred_mean) * lead_mask
                target_centered = (target_lead - target_mean) * lead_mask

                # Compute correlation
                numerator = (pred_centered * target_centered).sum()
                pred_std = torch.sqrt((pred_centered ** 2).sum() + self.smooth)
                target_std = torch.sqrt((target_centered ** 2).sum() + self.smooth)
                denominator = pred_std * target_std + self.smooth

                corr = numerator / denominator
                correlations.append(corr)

        if len(correlations) == 0:
            return torch.tensor(0.0, device=pred.device)

        # Average correlation, convert to loss (1 - correlation)
        avg_correlation = torch.stack(correlations).mean()
        return 1.0 - avg_correlation

    def _goldberger_loss(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for Goldberger equation violations.

        Goldberger equations:
        - III = II - I
        - aVR = -(I + II) / 2
        - aVL = I - II / 2
        - aVF = II - I / 2
        """
        batch_size = pred.shape[0]

        # Lead indices
        I_idx, II_idx, III_idx = 0, 1, 2
        aVR_idx, aVL_idx, aVF_idx = 3, 4, 5

        total_loss = torch.tensor(0.0, device=pred.device)

        for b in range(batch_size):
            # Get lead signals
            I = pred[b, I_idx]
            II = pred[b, II_idx]
            III = pred[b, III_idx]
            aVR = pred[b, aVR_idx]
            aVL = pred[b, aVL_idx]
            aVF = pred[b, aVF_idx]

            # Get masks - use minimum overlap for derived leads
            mask_I = mask[b, I_idx]
            mask_II = mask[b, II_idx]
            mask_III = mask[b, III_idx]
            mask_aVR = mask[b, aVR_idx]
            mask_aVL = mask[b, aVL_idx]
            mask_aVF = mask[b, aVF_idx]

            # Compute expected values from Goldberger equations
            # Note: These constraints apply where limb leads overlap (first 2.5s typically)
            III_expected = II - I
            aVR_expected = -(I + II) / 2
            aVL_expected = I - II / 2
            aVF_expected = II - I / 2

            # Combine masks for constraint regions
            mask_III_constraint = mask_I * mask_II * mask_III
            mask_aVR_constraint = mask_I * mask_II * mask_aVR
            mask_aVL_constraint = mask_I * mask_II * mask_aVL
            mask_aVF_constraint = mask_I * mask_II * mask_aVF

            # Compute violations (MSE where mask is valid)
            def masked_mse(pred_val: torch.Tensor, expected: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                count = m.sum() + self.smooth
                if count < 10:
                    return torch.tensor(0.0, device=pred.device)
                return ((pred_val - expected) ** 2 * m).sum() / count

            loss_III = masked_mse(III, III_expected, mask_III_constraint)
            loss_aVR = masked_mse(aVR, aVR_expected, mask_aVR_constraint)
            loss_aVL = masked_mse(aVL, aVL_expected, mask_aVL_constraint)
            loss_aVF = masked_mse(aVF, aVF_expected, mask_aVF_constraint)

            total_loss = total_loss + loss_III + loss_aVR + loss_aVL + loss_aVF

        return total_loss / batch_size


class CorrelationLoss(nn.Module):
    """Standalone Pearson correlation loss for signal similarity."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute 1 - Pearson correlation averaged over batch and leads.

        Args:
            pred: (batch, leads, length)
            target: (batch, leads, length)
            mask: Optional (batch, leads, length) binary mask

        Returns:
            Loss scalar
        """
        if mask is None:
            mask = torch.ones_like(target)

        batch_size, n_leads, _ = pred.shape
        correlations = []

        for b in range(batch_size):
            for lead in range(n_leads):
                m = mask[b, lead]
                valid_count = m.sum()

                if valid_count < 10:
                    continue

                p = pred[b, lead] * m
                t = target[b, lead] * m

                p_mean = p.sum() / valid_count
                t_mean = t.sum() / valid_count

                p_c = (p - p_mean) * m
                t_c = (t - t_mean) * m

                num = (p_c * t_c).sum()
                p_std = torch.sqrt((p_c ** 2).sum() + self.smooth)
                t_std = torch.sqrt((t_c ** 2).sum() + self.smooth)

                corr = num / (p_std * t_std + self.smooth)
                correlations.append(corr)

        if len(correlations) == 0:
            return torch.tensor(0.0, device=pred.device)

        return 1.0 - torch.stack(correlations).mean()


class SNRLoss(nn.Module):
    """Signal-to-noise ratio loss (negative SNR for minimization)."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute negative SNR as loss.

        SNR = 10 * log10(signal_power / noise_power)

        Args:
            pred: (batch, leads, length)
            target: (batch, leads, length)
            mask: Optional (batch, leads, length) binary mask

        Returns:
            Negative mean SNR in dB
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Signal power (target)
        signal_power = (target ** 2 * mask).sum() / (mask.sum() + self.smooth)

        # Noise power (difference)
        noise = (pred - target) * mask
        noise_power = (noise ** 2).sum() / (mask.sum() + self.smooth)

        # SNR in dB
        snr_db = 10 * torch.log10(signal_power / (noise_power + self.smooth) + self.smooth)

        # Return negative SNR (for minimization)
        return -snr_db


class CombinedSignalLoss(nn.Module):
    """
    Comprehensive loss combining multiple signal quality metrics.

    Useful for training with multiple loss components with configurable weights.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        correlation_weight: float = 0.5,
        snr_weight: float = 0.0,
        goldberger_weight: float = 0.1,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.snr_weight = snr_weight
        self.goldberger_weight = goldberger_weight

        self.signal_loss = SignalReconstructionLoss(
            mse_weight=1.0,
            correlation_weight=0.0,
            goldberger_weight=0.0,
            smooth=smooth,
        )
        self.correlation_loss = CorrelationLoss(smooth=smooth)
        self.snr_loss = SNRLoss(smooth=smooth) if snr_weight > 0 else None
        self.goldberger_loss = SignalReconstructionLoss(
            mse_weight=0.0,
            correlation_weight=0.0,
            goldberger_weight=1.0,
            smooth=smooth,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = torch.tensor(0.0, device=pred.device)

        if self.mse_weight > 0:
            total_loss = total_loss + self.mse_weight * self.signal_loss(pred, target, mask)

        if self.correlation_weight > 0:
            total_loss = total_loss + self.correlation_weight * self.correlation_loss(pred, target, mask)

        if self.snr_weight > 0 and self.snr_loss is not None:
            total_loss = total_loss + self.snr_weight * self.snr_loss(pred, target, mask)

        if self.goldberger_weight > 0:
            total_loss = total_loss + self.goldberger_weight * self.goldberger_loss(pred, target, mask)

        return total_loss
