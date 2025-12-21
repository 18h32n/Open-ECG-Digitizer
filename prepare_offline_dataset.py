"""
Prepare ECG Digitizer files for Kaggle Dataset upload.

This script creates a directory structure ready to upload as a Kaggle Dataset,
containing all source code and model weights needed for offline inference.

Usage:
    python prepare_offline_dataset.py

Output:
    ./kaggle_dataset/ directory with:
    ├── src/
    ├── weights/
    └── dataset-metadata.json
"""

import shutil
from pathlib import Path
import json

def create_dataset_structure():
    """Create Kaggle Dataset directory structure."""

    # Define paths
    project_root = Path(__file__).parent
    dataset_dir = project_root / "kaggle_dataset"

    print("=" * 60)
    print("ECG Digitizer - Kaggle Dataset Preparation")
    print("=" * 60)

    # Clean and create dataset directory
    if dataset_dir.exists():
        print(f"\n[*] Removing existing dataset directory...")
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(exist_ok=True)
    print(f"[OK] Created dataset directory: {dataset_dir}")

    # Copy source code
    print(f"\n[*] Copying source code...")
    src_dir = project_root / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, dataset_dir / "src",
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))

        # Count Python files
        py_files = list((dataset_dir / "src").rglob("*.py"))
        print(f"   [OK] Copied {len(py_files)} Python files")

        # Copy config files
        config_files = list((dataset_dir / "src").rglob("*.yml")) + \
                      list((dataset_dir / "src").rglob("*.yaml"))
        print(f"   [OK] Copied {len(config_files)} config files")
    else:
        print(f"   [ERROR] Source directory not found: {src_dir}")
        return False

    # Copy weights
    print(f"\n[*] Copying model weights...")
    weights_dir = project_root / "weights"
    if weights_dir.exists():
        shutil.copytree(weights_dir, dataset_dir / "weights",
                       ignore=shutil.ignore_patterns('*.log', '*.txt'))

        # List weights and sizes (both .pth and .pt extensions)
        weight_files = list((dataset_dir / "weights").glob("*.pth")) + \
                      list((dataset_dir / "weights").glob("*.pt"))
        total_size_mb = sum(f.stat().st_size for f in weight_files) / (1024 * 1024)

        print(f"   [OK] Copied {len(weight_files)} weight files:")
        for weight_file in weight_files:
            size_mb = weight_file.stat().st_size / (1024 * 1024)
            print(f"      - {weight_file.name}: {size_mb:.2f} MB")
        print(f"   [INFO] Total size: {total_size_mb:.2f} MB")
    else:
        print(f"   [ERROR] Weights directory not found: {weights_dir}")
        return False

    # Create dataset metadata for Kaggle
    print(f"\n[*] Creating dataset metadata...")
    metadata = {
        "title": "ECG Digitizer - Source Code and Weights",
        "id": "your-username/ecg-digitizer",  # User needs to update this
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["ecg", "medical", "computer-vision", "signal-processing"],
        "description": (
            "ECG Digitizer source code and trained model weights for offline Kaggle notebook execution. "
            "Includes UNet and LeadNet models for ECG grid detection and lead identification."
        ),
        "subtitle": "Pre-trained models and source code for ECG digitization"
    }

    metadata_path = dataset_dir / "dataset-metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"   [OK] Created: {metadata_path.name}")

    # Create README for dataset
    print(f"\n[*] Creating dataset README...")
    readme_content = """# ECG Digitizer - Offline Dataset

This dataset contains all the source code and pre-trained model weights needed to run ECG digitization in an offline Kaggle notebook.

## Contents

- `src/` - Complete source code for ECG digitization
  - `src/model/` - Model architectures (UNet, LeadNet)
  - `src/config/` - Configuration files
  - `src/kaggle_inference.py` - Main inference script

- `weights/` - Pre-trained model weights
  - `unet_best.pth` - UNet model for ECG grid detection
  - `leadnet_best.pth` - LeadNet model for lead identification

## Usage

1. **Attach this dataset to your Kaggle notebook**
   - In your notebook, go to "Add Data" → "Your Datasets" → select this dataset

2. **Run the offline baseline notebook**
   - Use the provided `kaggle_baseline_offline.ipynb` notebook
   - The notebook will automatically copy files from this dataset to the working directory

3. **Configure paths**
   - Update test data paths to point to your competition data
   - Run all cells to generate `submission.csv`

## Requirements

The inference code requires these packages (all available by default in Kaggle notebooks):
- numpy, scipy, scikit-learn
- pytorch, torchvision
- opencv-python
- pandas, matplotlib
- yacs (for configuration)

## Model Information

- **UNet**: Trained on ECG grid images for signal extraction
- **LeadNet**: Trained for ECG lead identification
- Both models use PyTorch and can run on CPU or GPU

## License

CC0-1.0 (Public Domain)
"""

    readme_path = dataset_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   [OK] Created: {readme_path.name}")

    # Print final instructions
    print("\n" + "=" * 60)
    print("[SUCCESS] Dataset preparation complete!")
    print("=" * 60)

    print(f"\nDataset directory: {dataset_dir.absolute()}")
    print(f"\nDataset contents:")
    print(f"   - {len(py_files)} Python source files")
    print(f"   - {len(weight_files)} model weight files ({total_size_mb:.2f} MB total)")
    print(f"   - Configuration files and metadata")

    print(f"\nNext steps to upload to Kaggle:")
    print(f"   1. Edit {metadata_path.name} and update 'id' to: your-kaggle-username/ecg-digitizer")
    print(f"   2. Install Kaggle CLI: pip install kaggle")
    print(f"   3. Ensure kaggle.json is configured (~/.kaggle/kaggle.json)")
    print(f"   4. Create new dataset:")
    print(f"      cd {dataset_dir.absolute()}")
    print(f"      kaggle datasets create -p .")
    print(f"   5. Or update existing dataset:")
    print(f"      kaggle datasets version -p . -m 'Updated weights and code'")

    print(f"\nAlternative: Manual upload via Kaggle website:")
    print(f"   1. Go to https://www.kaggle.com/datasets")
    print(f"   2. Click 'New Dataset'")
    print(f"   3. Upload the contents of: {dataset_dir.absolute()}")
    print(f"   4. Set title: 'ECG Digitizer - Source Code and Weights'")
    print(f"   5. Click 'Create'")

    print("\n" + "=" * 60)

    return True


def verify_dataset():
    """Verify the created dataset structure."""
    dataset_dir = Path(__file__).parent / "kaggle_dataset"

    required_files = [
        "src/kaggle_inference.py",
        "src/model/lead_identifier.py",
        "src/model/signal_extractor.py",
        "dataset-metadata.json",
        "README.md"
    ]

    # Check for weight files (either .pth or .pt extension)
    weight_files_required = ["weights/unet_weights_07072025.pt", "weights/lead_name_unet_weights_07072025.pt"]

    print("\n[*] Verifying dataset structure...")
    all_present = True

    # Check required files
    for required_file in required_files:
        file_path = dataset_dir / required_file
        if file_path.exists():
            print(f"   [OK] {required_file}")
        else:
            print(f"   [MISSING] {required_file}")
            all_present = False

    # Check weight files
    for weight_file in weight_files_required:
        file_path = dataset_dir / weight_file
        if file_path.exists():
            print(f"   [OK] {weight_file}")
        else:
            print(f"   [MISSING] {weight_file}")
            all_present = False

    if all_present:
        print("\n[SUCCESS] All required files present!")
    else:
        print("\n[WARNING] Some required files are missing!")

    return all_present


if __name__ == "__main__":
    success = create_dataset_structure()

    if success:
        verify_dataset()
        print("\n[SUCCESS] Dataset is ready for upload to Kaggle!")
    else:
        print("\n[ERROR] Dataset preparation failed! Please check the errors above.")
