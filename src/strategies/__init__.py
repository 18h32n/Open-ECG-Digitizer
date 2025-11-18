"""
Competition-winning strategies for ECG digitization.

This module contains advanced techniques to improve model performance:
- Physiological constraint enforcement
- Test-time augmentation (TTA)
- Synthetic data generation
- Model ensembling
"""

from src.strategies.physiological_constraints import (
    PhysiologicalConstraints,
    PhysiologicalLoss,
    apply_constraints_to_predictions,
)

from src.strategies.test_time_augmentation import (
    TestTimeAugmentation,
    EnsembleTTA,
    create_tta_inference_wrapper,
)

from src.strategies.synthetic_data_generator import (
    ECGImageGenerator,
    generate_synthetic_dataset,
)

__all__ = [
    # Physiological constraints
    "PhysiologicalConstraints",
    "PhysiologicalLoss",
    "apply_constraints_to_predictions",
    # Test-time augmentation
    "TestTimeAugmentation",
    "EnsembleTTA",
    "create_tta_inference_wrapper",
    # Synthetic data
    "ECGImageGenerator",
    "generate_synthetic_dataset",
]
