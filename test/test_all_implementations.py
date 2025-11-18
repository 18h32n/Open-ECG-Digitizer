"""
Comprehensive Test Suite for All 10 Novel ECG Digitization Ideas

This script tests all implementations to ensure they work correctly.
"""

import sys
import traceback
import numpy as np
import torch
from typing import Dict, List, Tuple


class TestResults:
    """Track test results."""

    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_result(self, name: str, passed: bool, error: str = None):
        """Add test result."""
        self.results[name] = {
            'passed': passed,
            'error': error
        }
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(f"{name}: {error}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        for name, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} - {name}")
            if result['error']:
                print(f"       Error: {result['error']}")

        print("\n" + "-"*80)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
        print("="*80)


def test_vision_transformer(results: TestResults):
    """Test Idea 1: Vision Transformer End-to-End."""
    print("\n🧪 Testing Idea 1: Vision Transformer End-to-End...")

    try:
        from src.model.vit2ecg import ViT2ECG, ViT2ECGWrapper, create_vit2ecg_model

        # Test model creation
        model = ViT2ECG(
            img_size=224,
            patch_size=16,
            num_leads=12,
            signal_length=5000,
            embed_dim=256,
            encoder_depth=2,
            decoder_depth=2,
            num_heads=4,
        )

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 12, 5000), f"Wrong output shape: {output.shape}"

        # Test wrapper
        wrapper = ViT2ECGWrapper(img_size=224, device='cpu')
        result = wrapper(x)

        assert 'canonical_lines' in result, "Missing canonical_lines in output"
        assert result['canonical_lines'].shape == (12, 5000), "Wrong canonical shape"

        results.add_result("Idea 1: Vision Transformer", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 1: Vision Transformer", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_lead_synthesis(results: TestResults):
    """Test Idea 2: Lead Synthesis from Rhythm."""
    print("\n🧪 Testing Idea 2: Lead Synthesis from Rhythm...")

    try:
        from src.strategies.lead_synthesis import LeadSynthesisNetwork, LeadSynthesizer

        # Test network
        model = LeadSynthesisNetwork(input_length=5000, hidden_dim=128, num_layers=2)

        # Test forward pass
        lead_ii = torch.randn(2, 5000)
        output = model(lead_ii)

        assert output.shape == (2, 11, 5000), f"Wrong output shape: {output.shape}"

        # Test synthesizer
        synthesizer = LeadSynthesizer(signal_length=5000)
        lead_ii_np = np.random.randn(5000)
        all_leads = synthesizer.synthesize_from_lead_ii(lead_ii_np)

        assert len(all_leads) == 12, "Should have 12 leads"
        assert 'II' in all_leads, "Should have Lead II"
        assert all_leads['I'].shape == (5000,), "Wrong signal shape"

        results.add_result("Idea 2: Lead Synthesis", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 2: Lead Synthesis", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_diffusion_denoiser(results: TestResults):
    """Test Idea 3: Diffusion Model Denoising."""
    print("\n🧪 Testing Idea 3: Diffusion Model Denoising...")

    try:
        from src.strategies.diffusion_denoiser import DiffusionUNet1D, ECGDiffusionDenoiser

        # Test U-Net
        model = DiffusionUNet1D(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            num_res_blocks=1,
            channel_mult=(1, 2),
            time_emb_dim=256,
        )

        # Test forward pass
        x = torch.randn(2, 1, 1000)
        t = torch.randint(0, 100, (2,))
        output = model(x, t)

        assert output.shape == (2, 1, 1000), f"Wrong output shape: {output.shape}"

        # Test denoiser
        denoiser = ECGDiffusionDenoiser(num_diffusion_steps=100, device='cpu')
        noisy_signal = np.random.randn(1000)
        denoised = denoiser.denoise_signal(noisy_signal, num_inference_steps=5)

        assert denoised.shape == (1000,), "Wrong denoised shape"

        results.add_result("Idea 3: Diffusion Denoising", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 3: Diffusion Denoising", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_ocr_layout_detector(results: TestResults):
    """Test Idea 4: OCR-Enhanced Layout Detection."""
    print("\n🧪 Testing Idea 4: OCR-Enhanced Layout Detection...")

    try:
        from src.strategies.ocr_layout_detector import (
            TextDetector, TextRecognizer, OCRLayoutDetector, TraceRegion
        )

        # Test text detector
        detector_model = TextDetector(pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        score_map = detector_model(x)

        assert score_map.shape[1] == 1, "Should output single channel"
        assert score_map.min() >= 0 and score_map.max() <= 1, "Should be in [0,1]"

        # Test text recognizer
        recognizer = TextRecognizer(img_height=32)
        x = torch.randn(1, 3, 32, 100)
        output = recognizer(x)

        assert output.shape[1] == 100, "Wrong sequence length"
        assert output.shape[2] == 37, "Wrong num classes"

        # Test OCR detector
        detector = OCRLayoutDetector(device='cpu')
        image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        bboxes = detector.detect_text_regions(image, threshold=0.5)

        assert isinstance(bboxes, list), "Should return list of bboxes"

        results.add_result("Idea 4: OCR Layout Detection", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 4: OCR Layout Detection", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_physics_calibration(results: TestResults):
    """Test Idea 5: Physics-Based Calibration."""
    print("\n🧪 Testing Idea 5: Physics-Based Calibration...")

    try:
        from src.strategies.physics_calibration import (
            CalibrationEstimator, StatisticalCalibrationValidator, PhysicsBasedCalibrator
        )

        # Test calibration estimator
        model = CalibrationEstimator(backbone='resnet34', pretrained=False)
        x = torch.randn(2, 3, 512, 512)
        predictions = model(x)

        assert 'mv_per_mm' in predictions, "Missing mv_per_mm"
        assert 'mm_per_pixel' in predictions, "Missing mm_per_pixel"
        assert 'confidence' in predictions, "Missing confidence"

        # Test validator
        validator = StatisticalCalibrationValidator()
        signal = np.random.randn(5000) * 100

        from src.strategies.physics_calibration import CalibrationParams
        cal = CalibrationParams(mv_per_pixel_y=0.01, mm_per_pixel_x=0.1, mm_per_pixel_y=0.1)

        is_valid, conf, metrics = validator.validate_calibration(signal, cal)
        assert isinstance(is_valid, bool), "Should return bool"
        assert 0 <= conf <= 1, "Confidence should be in [0,1]"

        # Test calibrator
        calibrator = PhysicsBasedCalibrator(device='cpu')
        image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        calibration = calibrator.estimate_calibration(image)

        assert calibration.mv_per_pixel_y > 0, "Should have positive calibration"

        results.add_result("Idea 5: Physics Calibration", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 5: Physics Calibration", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_adversarial_training(results: TestResults):
    """Test Idea 6: Adversarial Robustness Training."""
    print("\n🧪 Testing Idea 6: Adversarial Robustness Training...")

    try:
        from src.strategies.adversarial_training import FGSM, PGD, AutoAttack, AdversarialTrainer

        # Simple model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.fc = torch.nn.Linear(16, 12 * 100)

            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                x = self.fc(x)
                return x.view(-1, 12, 100)

        model = DummyModel()

        # Test FGSM
        fgsm = FGSM(epsilon=0.03)
        x = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 12, 100)
        loss_fn = torch.nn.MSELoss()

        adv_x = fgsm.attack(model, x, target, loss_fn)
        assert adv_x.shape == x.shape, "Adversarial example wrong shape"
        assert not torch.equal(adv_x, x), "Adversarial example should differ"

        # Test PGD
        pgd = PGD(epsilon=0.03, alpha=0.01, num_iter=5)
        adv_x_pgd = pgd.attack(model, x, target, loss_fn)
        assert adv_x_pgd.shape == x.shape, "PGD wrong shape"

        # Test AutoAttack
        auto = AutoAttack(epsilon=0.03)
        adv_examples = auto.attack(model, x[:1], target[:1], loss_fn)
        assert 'fgsm' in adv_examples, "Should have FGSM attack"
        assert 'pgd' in adv_examples, "Should have PGD attack"

        # Test trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = AdversarialTrainer(model, optimizer, device='cpu', attack_type='fgsm')

        losses = trainer.train_step(x, target, loss_fn)
        assert 'clean_loss' in losses, "Should have clean loss"
        assert 'adv_loss' in losses, "Should have adversarial loss"

        results.add_result("Idea 6: Adversarial Training", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 6: Adversarial Training", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_multi_hypothesis_grid(results: TestResults):
    """Test Idea 7: Multi-Hypothesis Grid Detection."""
    print("\n🧪 Testing Idea 7: Multi-Hypothesis Grid Detection...")

    try:
        from src.strategies.multi_hypothesis_grid import (
            GridDetector, MultiHypothesisGridDetector, GridHypothesis
        )

        # Create synthetic grid
        H, W = 500, 500
        grid = np.zeros((H, W))
        spacing = 50
        for x in range(0, W, spacing):
            grid[:, x] = 1.0
        for y in range(0, H, spacing):
            grid[y, :] = 1.0
        grid += np.random.rand(H, W) * 0.1

        # Test grid detector
        detector = GridDetector()
        hypotheses = detector.detect_grid_autocorrelation(grid)
        assert len(hypotheses) > 0, "Should detect at least one hypothesis"

        # Test multi-hypothesis detector
        multi_detector = MultiHypothesisGridDetector()
        all_hypotheses = multi_detector.generate_hypotheses(grid)
        assert len(all_hypotheses) >= 1, "Should generate hypotheses"

        # Test scoring
        best = multi_detector.select_best_hypothesis(all_hypotheses, grid)
        assert isinstance(best, GridHypothesis), "Should return GridHypothesis"
        assert best.confidence > 0, "Should have positive confidence"

        # Test ensemble
        ensembled = multi_detector.ensemble_hypotheses(all_hypotheses, grid, top_k=3)
        assert isinstance(ensembled, GridHypothesis), "Should return GridHypothesis"

        results.add_result("Idea 7: Multi-Hypothesis Grid", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 7: Multi-Hypothesis Grid", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_physiological_constraints(results: TestResults):
    """Test Idea 8: Physiological Constraints."""
    print("\n🧪 Testing Idea 8: Physiological Constraints...")

    try:
        from src.strategies.physiological_constraints import (
            PhysiologicalConstraints, PhysiologicalLoss, apply_constraints_to_predictions
        )

        # Test constraints
        constraints = PhysiologicalConstraints(alpha=0.3)

        # Create synthetic 12-lead ECG
        signals = torch.randn(12, 5000)

        # Apply Goldberger equations
        corrected = constraints.apply_goldberger_equations(signals)
        assert corrected.shape == signals.shape, "Shape should be preserved"

        # Test cross-lead optimization
        optimized = constraints.optimize_cross_lead_consistency(signals, iterations=3)
        assert optimized.shape == signals.shape, "Shape should be preserved"

        # Test heart rate validation
        hr_validation = constraints.validate_heart_rate_consistency(signals, fs=500.0)
        assert 'valid' in hr_validation, "Should have valid field"

        # Test amplitude check
        amp_check = constraints.check_amplitude_plausibility(signals)
        assert 'valid' in amp_check, "Should have valid field"

        # Test loss
        loss_fn = PhysiologicalLoss(weight=0.1)
        pred = torch.randn(2, 12, 5000)
        target = torch.randn(2, 12, 5000)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0, "Should return scalar loss"

        # Test convenience function
        signals_np = np.random.randn(12, 5000)
        corrected_np = apply_constraints_to_predictions(signals_np, fs=500.0, alpha=0.3)
        assert corrected_np.shape == signals_np.shape, "Shape should be preserved"

        results.add_result("Idea 8: Physiological Constraints", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 8: Physiological Constraints", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_test_time_augmentation(results: TestResults):
    """Test Idea 9: Test-Time Augmentation."""
    print("\n🧪 Testing Idea 9: Test-Time Augmentation...")

    try:
        from src.strategies.test_time_augmentation import TestTimeAugmentation

        # Simple model for testing
        class DummyModel:
            def __call__(self, image, layout_should_include_substring=None):
                signals = torch.randn(12, 5000)
                return {
                    'canonical_lines': signals,
                    'signal': {
                        'canonical_lines': signals,
                        'layout_matching_cost': 0.5,
                    },
                    'input_image': image.cpu(),
                }

        model = DummyModel()

        # Test TTA
        tta = TestTimeAugmentation(
            model=model,
            n_augmentations=3,
            flip_horizontal=True,
            flip_vertical=False,
            rotations=[-1, 0, 1],
            scales=[0.95, 1.0, 1.05],
        )

        # Test forward pass
        image = torch.randn(1, 3, 224, 224)
        result = tta(image)

        assert 'canonical_lines' in result, "Should have canonical_lines"
        assert 'tta_n_augmentations' in result, "Should have TTA count"
        assert result['canonical_lines'].shape == (12, 5000), "Wrong shape"

        results.add_result("Idea 9: Test-Time Augmentation", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 9: Test-Time Augmentation", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def test_synthetic_data_generator(results: TestResults):
    """Test Idea 10: Synthetic Data Generator."""
    print("\n🧪 Testing Idea 10: Synthetic Data Generator...")

    try:
        from src.strategies.synthetic_data_generator import ECGImageGenerator

        # Test generator
        generator = ECGImageGenerator(
            image_size=(1000, 800),
            grid_spacing_mm=5.0,
            mm_per_pixel=0.1,
        )

        # Test grid generation
        grid = generator.generate_grid(800, 600)
        assert grid.shape == (600, 800, 3), f"Wrong grid shape: {grid.shape}"
        assert grid.dtype == np.uint8, "Should be uint8"

        # Test signal drawing
        signal = np.sin(np.linspace(0, 10, 1000)) * 0.5
        image = generator.draw_signal(grid, signal, start_x=10, start_y=300)
        assert image.shape == grid.shape, "Shape should match grid"

        # Test degradations
        clean_image = np.ones((500, 500, 3), dtype=np.uint8) * 255

        degraded = generator.apply_degradations(clean_image, 'scanned_color', severity=0.5)
        assert degraded.shape == clean_image.shape, "Shape should be preserved"

        degraded = generator.apply_degradations(clean_image, 'photo', severity=0.3)
        assert degraded.shape == clean_image.shape, "Shape should be preserved"

        degraded = generator.apply_degradations(clean_image, 'stained', severity=0.7)
        assert degraded.shape == clean_image.shape, "Shape should be preserved"

        # Test training pair generation
        signals = {
            'I': np.random.randn(5000) * 0.5,
            'II': np.random.randn(5000) * 0.5,
            'III': np.random.randn(5000) * 0.5,
            'aVR': np.random.randn(5000) * 0.5,
            'aVL': np.random.randn(5000) * 0.5,
            'aVF': np.random.randn(5000) * 0.5,
            'V1': np.random.randn(5000) * 0.5,
            'V2': np.random.randn(5000) * 0.5,
            'V3': np.random.randn(5000) * 0.5,
            'V4': np.random.randn(5000) * 0.5,
            'V5': np.random.randn(5000) * 0.5,
            'V6': np.random.randn(5000) * 0.5,
        }

        image, signals_out = generator.generate_training_pair(signals, degradation_type='clean')
        assert isinstance(image, np.ndarray), "Should return numpy array"
        assert image.shape[2] == 3, "Should be RGB"

        results.add_result("Idea 10: Synthetic Data Generator", True)
        print("   ✅ All tests passed!")

    except Exception as e:
        results.add_result("Idea 10: Synthetic Data Generator", False, str(e))
        print(f"   ❌ Test failed: {e}")
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE FOR ALL 10 NOVEL IDEAS")
    print("="*80)

    results = TestResults()

    # Run all tests
    test_vision_transformer(results)
    test_lead_synthesis(results)
    test_diffusion_denoiser(results)
    test_ocr_layout_detector(results)
    test_physics_calibration(results)
    test_adversarial_training(results)
    test_multi_hypothesis_grid(results)
    test_physiological_constraints(results)
    test_test_time_augmentation(results)
    test_synthetic_data_generator(results)

    # Print summary
    results.print_summary()

    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
