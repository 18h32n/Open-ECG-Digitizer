"""
Adversarial Robustness Training

Improve model robustness by:
1. Generating adversarial examples that fool the current model
2. Retraining on adversarial examples
3. Iterative adversarial training loop

This helps identify and fix model vulnerabilities, especially for
difficult cases like damaged/stained ECGs.

Expected gain: +5-10 dB SNR on difficult cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Dict, Tuple


class FGSM:
    """Fast Gradient Sign Method for generating adversarial examples."""

    def __init__(self, epsilon: float = 0.03):
        """
        Args:
            epsilon: Perturbation magnitude
        """
        self.epsilon = epsilon

    def attack(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Generate adversarial example using FGSM.

        Args:
            model: Target model
            image: Input image (B, C, H, W)
            target: Target output (B, ...)
            loss_fn: Loss function

        Returns:
            Adversarial image
        """
        image = image.clone().detach().requires_grad_(True)

        # Forward pass
        output = model(image)

        # Compute loss
        if isinstance(output, dict):
            output = output.get('canonical_lines', output.get('signal', {}).get('canonical_lines'))

        loss = loss_fn(output, target)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Generate adversarial example
        data_grad = image.grad.data

        # FGSM: add epsilon * sign(gradient)
        perturbed_image = image + self.epsilon * data_grad.sign()

        # Clip to valid range [0, 1]
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image.detach()


class PGD:
    """Projected Gradient Descent - stronger than FGSM."""

    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 10,
    ):
        """
        Args:
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def attack(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Generate adversarial example using PGD.

        Args:
            model: Target model
            image: Input image
            target: Target output
            loss_fn: Loss function

        Returns:
            Adversarial image
        """
        original_image = image.clone().detach()

        # Start with random perturbation
        perturbed_image = image + torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        for _ in range(self.num_iter):
            perturbed_image = perturbed_image.clone().detach().requires_grad_(True)

            # Forward pass
            output = model(perturbed_image)

            if isinstance(output, dict):
                output = output.get('canonical_lines', output.get('signal', {}).get('canonical_lines'))

            loss = loss_fn(output, target)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Update perturbation
            data_grad = perturbed_image.grad.data
            perturbed_image = perturbed_image + self.alpha * data_grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(perturbed_image - original_image, -self.epsilon, self.epsilon)
            perturbed_image = torch.clamp(original_image + delta, 0, 1)

        return perturbed_image.detach()


class AutoAttack:
    """Combines multiple attacks for comprehensive adversarial testing."""

    def __init__(self, epsilon: float = 0.03):
        self.epsilon = epsilon
        self.attacks = {
            'fgsm': FGSM(epsilon=epsilon),
            'pgd': PGD(epsilon=epsilon, alpha=epsilon/4, num_iter=10),
            'pgd_strong': PGD(epsilon=epsilon, alpha=epsilon/10, num_iter=40),
        }

    def attack(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Run multiple attacks and return all adversarial examples.

        Args:
            model: Target model
            image: Input image
            target: Target output
            loss_fn: Loss function

        Returns:
            Dictionary of adversarial examples
        """
        adversarial_examples = {}

        for attack_name, attacker in self.attacks.items():
            adv_image = attacker.attack(model, image, target, loss_fn)
            adversarial_examples[attack_name] = adv_image

        return adversarial_examples


class AdversarialTrainer:
    """High-level adversarial training interface."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        epsilon: float = 0.03,
        alpha: float = 0.7,  # Weight for adversarial loss
        attack_type: str = 'pgd',
    ):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device
            epsilon: Perturbation magnitude
            alpha: Weight for adversarial examples (1-alpha for clean)
            attack_type: Type of attack ('fgsm', 'pgd', 'auto')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha

        # Select attack
        if attack_type == 'fgsm':
            self.attacker = FGSM(epsilon=epsilon)
        elif attack_type == 'pgd':
            self.attacker = PGD(epsilon=epsilon, alpha=epsilon/4, num_iter=10)
        elif attack_type == 'auto':
            self.attacker = AutoAttack(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    def train_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """Single adversarial training step.

        Args:
            images: Batch of images
            targets: Batch of targets
            loss_fn: Loss function

        Returns:
            Dictionary of losses
        """
        images = images.to(self.device)
        targets = targets.to(self.device)

        # 1. Clean loss
        self.model.train()
        clean_output = self.model(images)

        if isinstance(clean_output, dict):
            clean_output = clean_output.get('canonical_lines', clean_output.get('signal', {}).get('canonical_lines'))

        clean_loss = loss_fn(clean_output, targets)

        # 2. Generate adversarial examples
        self.model.eval()  # Fix batch norm for attack generation

        if isinstance(self.attacker, AutoAttack):
            # Use strongest attack
            adv_examples = self.attacker.attack(self.model, images, targets, loss_fn)
            adv_images = adv_examples['pgd_strong']
        else:
            adv_images = self.attacker.attack(self.model, images, targets, loss_fn)

        # 3. Adversarial loss
        self.model.train()
        adv_output = self.model(adv_images)

        if isinstance(adv_output, dict):
            adv_output = adv_output.get('canonical_lines', adv_output.get('signal', {}).get('canonical_lines'))

        adv_loss = loss_fn(adv_output, targets)

        # 4. Combined loss
        total_loss = (1 - self.alpha) * clean_loss + self.alpha * adv_loss

        # 5. Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'clean_loss': clean_loss.item(),
            'adv_loss': adv_loss.item(),
        }

    def evaluate_robustness(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
        metric_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Evaluate model robustness against adversarial attacks.

        Args:
            images: Batch of images
            targets: Batch of targets
            loss_fn: Loss function
            metric_fn: Optional metric function (e.g., SNR)

        Returns:
            Dictionary of metrics
        """
        images = images.to(self.device)
        targets = targets.to(self.device)

        self.model.eval()

        results = {}

        # Clean performance
        with torch.no_grad():
            clean_output = self.model(images)
            if isinstance(clean_output, dict):
                clean_output = clean_output.get('canonical_lines', clean_output.get('signal', {}).get('canonical_lines'))

            results['clean_loss'] = loss_fn(clean_output, targets).item()

            if metric_fn:
                results['clean_metric'] = metric_fn(clean_output, targets).item()

        # Adversarial performance
        if isinstance(self.attacker, AutoAttack):
            adv_examples = self.attacker.attack(self.model, images, targets, loss_fn)

            for attack_name, adv_images in adv_examples.items():
                with torch.no_grad():
                    adv_output = self.model(adv_images)
                    if isinstance(adv_output, dict):
                        adv_output = adv_output.get('canonical_lines', adv_output.get('signal', {}).get('canonical_lines'))

                    results[f'{attack_name}_loss'] = loss_fn(adv_output, targets).item()

                    if metric_fn:
                        results[f'{attack_name}_metric'] = metric_fn(adv_output, targets).item()
        else:
            adv_images = self.attacker.attack(self.model, images, targets, loss_fn)

            with torch.no_grad():
                adv_output = self.model(adv_images)
                if isinstance(adv_output, dict):
                    adv_output = adv_output.get('canonical_lines', adv_output.get('signal', {}).get('canonical_lines'))

                results['adv_loss'] = loss_fn(adv_output, targets).item()

                if metric_fn:
                    results['adv_metric'] = metric_fn(adv_output, targets).item()

        return results


class DefensiveDistillation:
    """Train a distilled model that's more robust to adversarial examples."""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 20.0,
        device: str = 'cuda',
    ):
        """
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            temperature: Softening temperature for distillation
            device: Device
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.temperature = temperature
        self.device = device

    def distill_step(
        self,
        images: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single distillation training step.

        Args:
            images: Batch of images
            optimizer: Optimizer for student

        Returns:
            Distillation loss
        """
        images = images.to(self.device)

        # Get teacher soft labels
        with torch.no_grad():
            teacher_output = self.teacher(images)
            if isinstance(teacher_output, dict):
                teacher_output = teacher_output.get('canonical_lines', teacher_output.get('signal', {}).get('canonical_lines'))

            # Apply temperature
            teacher_soft = F.softmax(teacher_output / self.temperature, dim=-1)

        # Get student predictions
        student_output = self.student(images)
        if isinstance(student_output, dict):
            student_output = student_output.get('canonical_lines', student_output.get('signal', {}).get('canonical_lines'))

        student_soft = F.log_softmax(student_output / self.temperature, dim=-1)

        # KL divergence loss
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        loss = loss * (self.temperature ** 2)  # Scale by T^2

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


def find_worst_case_examples(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: str = 'cuda',
    top_k: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find worst-case examples where model performs poorly.

    Useful for identifying failure modes and creating hard negative sets.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        loss_fn: Loss function
        device: Device
        top_k: Number of worst examples to return

    Returns:
        (worst_images, worst_targets, worst_losses)
    """
    model.eval()

    all_images = []
    all_targets = []
    all_losses = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get('canonical_lines', outputs.get('signal', {}).get('canonical_lines'))

            # Compute per-sample losses
            for i in range(len(images)):
                loss = loss_fn(outputs[i:i+1], targets[i:i+1])
                all_images.append(images[i])
                all_targets.append(targets[i])
                all_losses.append(loss.item())

    # Sort by loss (worst first)
    indices = np.argsort(all_losses)[::-1][:top_k]

    worst_images = torch.stack([all_images[i] for i in indices])
    worst_targets = torch.stack([all_targets[i] for i in indices])
    worst_losses = torch.tensor([all_losses[i] for i in indices])

    return worst_images, worst_targets, worst_losses


if __name__ == '__main__':
    print("Testing Adversarial Training...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.fc = nn.Linear(64, 12 * 5000)

        def forward(self, x):
            x = self.conv(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = self.fc(x)
            return x.view(-1, 12, 5000)

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create trainer
    trainer = AdversarialTrainer(
        model=model,
        optimizer=optimizer,
        device='cpu',  # Use CPU for testing
        epsilon=0.03,
        alpha=0.7,
        attack_type='fgsm',
    )

    # Test data
    images = torch.randn(2, 3, 224, 224)
    targets = torch.randn(2, 12, 5000)
    loss_fn = nn.MSELoss()

    # Training step
    losses = trainer.train_step(images, targets, loss_fn)

    print(f"Training losses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    # Evaluation
    metrics = trainer.evaluate_robustness(images, targets, loss_fn)

    print(f"\nRobustness metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n✅ Adversarial Training ready!")
