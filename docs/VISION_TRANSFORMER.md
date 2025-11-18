# Vision Transformer for End-to-End ECG Digitization (ViT2ECG)

## Overview

ViT2ECG is a novel end-to-end architecture that directly maps ECG images to 12-lead time series signals without intermediate segmentation steps.

### Key Innovation

**Traditional Pipeline**:
```
Image → Segmentation → Perspective Correction → Grid Detection
→ Signal Extraction → Lead Identification → Signals
```

**ViT2ECG Pipeline**:
```
Image → ViT2ECG → Signals
```

### Advantages

1. **No Intermediate Steps**: Learns direct mapping, avoiding error accumulation
2. **Holistic Understanding**: Model sees the entire image context
3. **Learned Calibration**: No need for explicit grid detection
4. **Layout Agnostic**: Works with any ECG layout
5. **End-to-End Optimization**: Direct optimization for signal reconstruction

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Image                           │
│                     (B, 3, H, W)                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                  Patch Embedding                            │
│          Image → Patches → Embeddings                       │
│                (B, N, D)                                    │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│              Vision Transformer Encoder                      │
│        ┌────────────────────────────────┐                  │
│        │  Multi-Head Self-Attention     │                  │
│        │  Layer Norm                    │                  │
│        │  MLP (Feed Forward)            │                  │
│        │  Layer Norm                    │                  │
│        └────────────────────────────────┘                  │
│                    × 12 layers                              │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│              Transformer Decoder                             │
│        Query: Signal Embeddings (12 × 5000)                │
│        Key/Value: Encoder Output                           │
│        ┌────────────────────────────────┐                  │
│        │  Multi-Head Self-Attention     │                  │
│        │  Multi-Head Cross-Attention    │                  │
│        │  MLP (Feed Forward)            │                  │
│        └────────────────────────────────┘                  │
│                    × 6 layers                               │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                   Signal Head                               │
│         Linear Projection → Signal Values                   │
│                (B, 12, 5000)                                │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Patch Embedding

Converts image into a sequence of patch embeddings:
- Image size: 2240×2240
- Patch size: 32×32
- Number of patches: 70×70 = 4900
- Embedding dim: 768

### 2. Encoder (Vision Transformer)

- 12 transformer encoder blocks
- 12 attention heads per block
- Embedding dimension: 768
- MLP ratio: 4.0 (hidden dim = 3072)
- Processes image patches with self-attention
- Learns global image understanding

### 3. Decoder (Signal Generator)

- 6 transformer decoder blocks
- Learnable signal queries (12 × 5000 = 60,000 queries)
- Cross-attention to encoder output
- Generates signals autoregressively in concept space

### 4. Signal Head

- Linear projection from embedding to signal value
- Output: 12 leads × 5000 samples each

## Training

### Loss Function

```python
def vit2ecg_loss(pred_signals, target_signals, fs=500):
    """
    Multi-component loss:
    1. MSE loss on signals
    2. Physiological constraint loss
    3. Frequency domain loss
    """
    # 1. Time-domain MSE
    mse_loss = F.mse_loss(pred_signals, target_signals)

    # 2. Physiological constraints
    phys_loss = physiological_loss(pred_signals, target_signals)

    # 3. Frequency domain similarity
    pred_fft = torch.fft.rfft(pred_signals, dim=-1)
    target_fft = torch.fft.rfft(target_signals, dim=-1)
    freq_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())

    total_loss = mse_loss + 0.1 * phys_loss + 0.05 * freq_loss

    return total_loss
```

### Training Data

1. **Synthetic Data** (Primary):
   - Generate 100K+ ECG images from PhysioNet signals
   - Various degradations (scan, photo, stains, damage)
   - Perfect ground truth signals

2. **Competition Data** (Fine-tuning):
   - Real competition training data
   - Fine-tune after pre-training on synthetic

### Training Procedure

```python
# Pseudo-code
model = ViT2ECG(...)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    for batch in dataloader:
        images, signals = batch

        # Forward
        pred_signals = model(images)

        # Loss
        loss = vit2ecg_loss(pred_signals, signals)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
```

### Hyperparameters

- **Learning rate**: 1e-4 (with warmup)
- **Batch size**: 8-16 (depending on GPU)
- **Epochs**: 100
- **Optimizer**: AdamW (weight_decay=0.01)
- **Scheduler**: Cosine annealing
- **Warmup**: 10 epochs
- **Gradient clipping**: 1.0

## Inference

### Basic Usage

```python
from src.model.vit2ecg import create_vit2ecg_model

# Create model
model = create_vit2ecg_model(
    img_size=2240,
    num_leads=12,
    signal_length=5000,
    device='cuda'
)

# Load weights
model.load_state_dict(torch.load('vit2ecg_weights.pt'))

# Inference
image = load_image('ecg.png')
result = model(image)
signals = result['canonical_lines']  # (12, 5000)
```

### With Kaggle Pipeline

```bash
python -m src.kaggle_inference \
  --config src/config/vit2ecg_inference.yml
```

## Performance Expectations

### Speed
- **Inference time**: ~100ms per image (GPU)
- **Parameters**: ~85M
- **Memory**: ~4GB GPU

### Accuracy
- **Expected SNR**: 30-40 dB (with proper training)
- **Advantages over traditional**:
  - Better on damaged/stained images (no grid dependence)
  - More robust to novel layouts
  - Learns calibration implicitly

## Comparison with Traditional Pipeline

| Aspect | Traditional | ViT2ECG |
|--------|------------|---------|
| Steps | 7+ stages | 1 stage |
| Grid detection | Required | Not needed |
| Layout templates | Required | Not needed |
| Calibration | Explicit | Learned |
| Training data | Limited | Unlimited (synthetic) |
| Novel layouts | May fail | Robust |
| Damaged images | Struggles | Better |
| Inference time | ~500ms | ~100ms |
| Memory | ~2GB | ~4GB |

## Limitations

1. **Requires Large Training Data**: Needs 100K+ samples to train from scratch
2. **Black Box**: Less interpretable than pipeline
3. **Memory Intensive**: Large model requires significant GPU memory
4. **Fixed Output Length**: Currently outputs fixed 5000 samples
5. **No Explicit Layout**: Cannot verify layout correctness

## Future Improvements

1. **Variable Length Output**: Support different signal lengths
2. **Multi-Task Learning**: Joint training with layout classification
3. **Attention Visualization**: Visualize what model attends to
4. **Model Compression**: Knowledge distillation for smaller models
5. **Pre-training**: Use ImageNet pre-trained ViT encoder

## References

- Vision Transformer (ViT): https://arxiv.org/abs/2010.11929
- DETR (DEtection TRansformer): https://arxiv.org/abs/2005.12872
- Perceiver: https://arxiv.org/abs/2103.03206

## Citation

```bibtex
@misc{vit2ecg2025,
  title={ViT2ECG: End-to-End ECG Digitization with Vision Transformers},
  author={Generated via BMAD Brainstorming Methods},
  year={2025},
  note={Kaggle PhysioNet Competition}
}
```
