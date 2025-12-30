"""
Kaggle Training Notebook Script
Copy-paste these cells into your Kaggle notebook after running kaggle_setup.py cells 1-4

Prerequisites:
- Run kaggle_setup.py cells 1-4 first (clone, install deps, download weights)
- Competition data attached to notebook
- GPU accelerator enabled in Kaggle settings!

Expected performance with P100/T4:
- Phase 1 (frozen UNet): ~8 hours, 10 epochs
- Phase 2 (full fine-tune): ~7 hours, 5 epochs
"""

# ============================================================================
# CELL T1: Import Training Modules (After kaggle_setup.py cells 1-4)
# ============================================================================
import os
import sys

# Fix matplotlib backend
if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']
os.environ['MPLBACKEND'] = 'Agg'

# Add repo to path
os.chdir('/kaggle/working/Open-ECG-Digitizer')
sys.path.insert(0, '/kaggle/working/Open-ECG-Digitizer')

# Run in venv context
VENV_PYTHON = '/kaggle/working/venv/bin/python'

print("Environment ready for training!")

# ============================================================================
# CELL T2: Verify Competition Data and GPU
# ============================================================================
# Check GPU first!
!{VENV_PYTHON} -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - Enable GPU in notebook settings!')"

# Check competition data is attached
TRAIN_DIR = '/kaggle/input/physionet-ecg-image-digitization/train'
TRAIN_CSV = '/kaggle/input/physionet-ecg-image-digitization/train.csv'

if os.path.exists(TRAIN_DIR):
    sample_dirs = os.listdir(TRAIN_DIR)[:5]
    print(f"Train directory found with {len(os.listdir(TRAIN_DIR))} samples")
    print(f"Sample IDs: {sample_dirs}")
else:
    print("ERROR: Train directory not found!")
    print("Make sure you've attached the competition dataset to your notebook")

if os.path.exists(TRAIN_CSV):
    import pandas as pd
    df = pd.read_csv(TRAIN_CSV)
    print(f"Train CSV found: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample frequencies: {df['fs'].unique()}")
else:
    print("ERROR: Train CSV not found!")

# ============================================================================
# CELL T3: Quick Dataset Test
# ============================================================================
# Write test script to file to avoid escaping issues
with open('/kaggle/working/test_dataset.py', 'w') as f:
    f.write('''
import sys
sys.path.insert(0, '/kaggle/working/Open-ECG-Digitizer')

from src.dataset.kaggle_ecg import KaggleECGDataset

# Create small test dataset
test_dataset = KaggleECGDataset(
    train_dir='/kaggle/input/physionet-ecg-image-digitization/train',
    target_length=5000,
    degradation_types=['0001', '0005'],
    cache_signals=True,
    split='train',
    val_ratio=0.1,
)

print(f"Dataset size: {len(test_dataset)}")

# Test loading a sample
image, signals, lead_mask, metadata = test_dataset[0]
print(f"Image shape: {image.shape}")
print(f"Signals shape: {signals.shape}")
print(f"Lead mask shape: {lead_mask.shape}")
print(f"Metadata: {metadata}")

import torch
valid_signals = signals[lead_mask > 0]
print(f"Signal range: [{valid_signals.min():.3f}, {valid_signals.max():.3f}] mV")
print(f"Signal mean: {valid_signals.mean():.3f} mV")
print("Dataset test passed!")
''')

!{VENV_PYTHON} /kaggle/working/test_dataset.py

# ============================================================================
# CELL T4: Start Training - Phase 1 (Frozen UNet)
# ============================================================================
# Write training script to file
with open('/kaggle/working/run_phase1.py', 'w') as f:
    f.write('''
import sys
import os
sys.path.insert(0, '/kaggle/working/Open-ECG-Digitizer')
os.chdir('/kaggle/working/Open-ECG-Digitizer')

from src.kaggle_train import train

print("Starting Phase 1 Training (Frozen UNet)...")
print("This will take several hours on GPU")

history = train(
    train_dir='/kaggle/input/physionet-ecg-image-digitization/train',
    output_dir='/kaggle/working/checkpoints',
    unet_weights_path='./weights/unet_weights_07072025.pt',
    epochs=10,
    batch_size=2,
    learning_rate=1e-4,
    freeze_unet_epochs=10,
    use_amp=True,
    checkpoint_freq=2,
    degradation_types=['0001', '0003', '0005'],
)

print("Phase 1 Training History:")
for i, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
    print(f"  Epoch {i+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
''')

print("Starting Phase 1 Training (Frozen UNet)...")
print("Estimated time: ~8 hours for 10 epochs on GPU")
print("Training on: Clean + Scanned + Mobile photos")
!{VENV_PYTHON} /kaggle/working/run_phase1.py

# ============================================================================
# CELL T5: Start Training - Phase 2 (Full Fine-tune) - OPTIONAL
# ============================================================================
# Only run this after Phase 1 completes successfully
with open('/kaggle/working/run_phase2.py', 'w') as f:
    f.write('''
import sys
import os
sys.path.insert(0, '/kaggle/working/Open-ECG-Digitizer')
os.chdir('/kaggle/working/Open-ECG-Digitizer')

from src.kaggle_train import train

print("Starting Phase 2 Training (Full Fine-tune)...")

history = train(
    train_dir='/kaggle/input/physionet-ecg-image-digitization/train',
    output_dir='/kaggle/working/checkpoints_phase2',
    unet_weights_path=None,
    epochs=5,
    batch_size=2,
    learning_rate=1e-5,
    freeze_unet_epochs=0,
    use_amp=True,
    checkpoint_freq=1,
    resume_from='/kaggle/working/checkpoints/best_model.pt',
    degradation_types=None,
)

print("Phase 2 Training History:")
for i, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
    print(f"  Epoch {i+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
''')

print("Starting Phase 2 Training (Full Fine-tune)...")
print("Estimated time: ~7 hours for 5 epochs")
!{VENV_PYTHON} /kaggle/working/run_phase2.py

# ============================================================================
# CELL T6: Save Final Model for Inference
# ============================================================================
with open('/kaggle/working/save_model.py', 'w') as f:
    f.write('''
import sys
import os
import torch
sys.path.insert(0, '/kaggle/working/Open-ECG-Digitizer')

# Find best checkpoint
checkpoint_path = '/kaggle/working/checkpoints_phase2/best_model.pt'
if not os.path.exists(checkpoint_path):
    checkpoint_path = '/kaggle/working/checkpoints/best_model.pt'

if not os.path.exists(checkpoint_path):
    print("ERROR: No checkpoint found!")
    sys.exit(1)

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract UNet weights
unet_weights = {}
signal_head_weights = {}

for key, value in checkpoint['model_state_dict'].items():
    if key.startswith('unet.'):
        unet_weights[key[5:]] = value
    elif key.startswith('signal_head.'):
        signal_head_weights[key[12:]] = value

# Save weights
unet_save_path = '/kaggle/working/trained_unet_weights.pt'
torch.save(unet_weights, unet_save_path)
print(f"Saved trained UNet weights to {unet_save_path}")

signal_head_save_path = '/kaggle/working/signal_head_weights.pt'
torch.save(signal_head_weights, signal_head_save_path)
print(f"Saved signal head weights to {signal_head_save_path}")

full_save_path = '/kaggle/working/full_trained_model.pt'
torch.save(checkpoint['model_state_dict'], full_save_path)
print(f"Saved full model to {full_save_path}")

print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
print(f"Training epochs completed: {checkpoint['epoch'] + 1}")
''')

print("Saving trained models...")
!{VENV_PYTHON} /kaggle/working/save_model.py

# ============================================================================
# CELL T7: Run Inference with Trained Model
# ============================================================================
# Copy trained weights to expected location for inference
import shutil
trained_weights = '/kaggle/working/trained_unet_weights.pt'
if os.path.exists(trained_weights):
    shutil.copy(trained_weights, '/kaggle/working/Open-ECG-Digitizer/weights/unet_weights_07072025.pt')
    print("Copied trained weights to inference location")

# Run inference
device = 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu'
print(f"Running inference on {device}...")

!{VENV_PYTHON} -m src.kaggle_inference \
    --config src/config/kaggle_inference.yml \
    DATA.test_csv_path=/kaggle/input/physionet-ecg-image-digitization/test.csv \
    DATA.test_images_dir=/kaggle/input/physionet-ecg-image-digitization/test \
    DATA.submission_path=/kaggle/working/submission.csv \
    MODEL.KWARGS.device={device}

print("Inference complete! Submission saved to /kaggle/working/submission.csv")

# ============================================================================
# CELL T8: Validate Submission
# ============================================================================
with open('/kaggle/working/validate_submission.py', 'w') as f:
    f.write('''
import pandas as pd
import sys

try:
    submission = pd.read_csv('/kaggle/working/submission.csv')
except FileNotFoundError:
    print("ERROR: submission.csv not found!")
    sys.exit(1)

print(f"Submission shape: {submission.shape}")
print(f"Columns: {list(submission.columns)}")
print(f"First 10 rows:")
print(submission.head(10))

nan_count = submission['value'].isna().sum()
nan_pct = nan_count / len(submission) * 100
print(f"NaN values: {nan_count} ({nan_pct:.2f}%)")

print(f"Value statistics:")
print(f"  Min: {submission['value'].min():.4f}")
print(f"  Max: {submission['value'].max():.4f}")
print(f"  Mean: {submission['value'].mean():.4f}")
print(f"  Std: {submission['value'].std():.4f}")

if nan_pct > 20:
    print("WARNING: High NaN percentage. Model may need more training.")
else:
    print("Submission looks good!")
''')

!{VENV_PYTHON} /kaggle/working/validate_submission.py

# ============================================================================
# CELL T9: Submit to Kaggle (Optional)
# ============================================================================
print("To submit to Kaggle, run:")
print("!kaggle competitions submit -c physionet-ecg-image-digitization -f /kaggle/working/submission.csv -m 'Trained model submission'")
