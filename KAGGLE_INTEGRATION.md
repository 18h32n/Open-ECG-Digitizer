# Kaggle PhysioNet ECG Digitization Competition Integration

This document explains how to use the Open-ECG-Digitizer model for the [Kaggle PhysioNet ECG Image Digitization Competition](https://www.kaggle.com/competitions/physionet-ecg-image-digitization).

## Overview

The integration script (`src/kaggle_inference.py`) adapts the Open-ECG-Digitizer model output to match the competition requirements:

- **Lead II**: 10 seconds duration (`floor(fs * 10)` samples)
- **Other 11 leads**: 2.5 seconds duration (`floor(fs * 2.5)` samples)
- **Output units**: Millivolts (mV) - converted from the model's microvolts (µV)
- **Output format**: Long format CSV with `id,value` pairs

## Competition Details

### Evaluation Metric

The competition uses a modified Signal-to-Noise Ratio (SNR):

1. **Time alignment**: Optimal horizontal shift (max 0.2 seconds) via cross-correlation
2. **Vertical alignment**: Removes constant amplitude offset
3. **SNR calculation**: `10 * log10(signal_power / noise_power)` in dB
4. Powers are summed across all 12 leads before computing SNR per record
5. Final score: Average SNR across all test records

### Data Format

**Input**:
- ECG images with various quality issues (scanned, photographed, damaged, etc.)
- `test.csv` with metadata: `id`, `fs` (sampling frequency), `number_of_rows`

**Output**:
- `submission.csv` with format: `id,value`
- Where `id = {base_id}_{row_id}_{lead}`

## Setup

### 1. Install Dependencies

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights

```bash
git lfs pull
```

This downloads:
- `weights/unet_weights_07072025.pt` - Semantic segmentation model
- `weights/lead_name_unet_weights_07072025.pt` - Lead name identification model

### 3. Prepare Competition Data

Download the competition data from Kaggle and organize as follows:

```
/kaggle/input/physionet-ecg-image-digitization/
├── test.csv
├── test/
│   ├── {id}.png
│   ├── {id}.png
│   └── ...
└── sample_submission.parquet
```

## Usage

### Basic Usage

Run inference with the default configuration:

```bash
python -m src.kaggle_inference --config src/config/kaggle_inference.yml
```

### Custom Paths

Override configuration options via command line:

```bash
python -m src.kaggle_inference \
  --config src/config/kaggle_inference.yml \
  DATA.test_csv_path=/path/to/test.csv \
  DATA.test_images_dir=/path/to/test \
  DATA.submission_path=/path/to/submission.csv
```

### Using CPU Instead of GPU

```bash
python -m src.kaggle_inference \
  --config src/config/kaggle_inference.yml \
  MODEL.KWARGS.device='cpu' \
  MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device='cpu'
```

## Configuration

The configuration file `src/config/kaggle_inference.yml` contains:

### Model Settings

- **device**: 'cuda' or 'cpu'
- **resample_size**: Maximum image dimension for processing (default: 3000)
- **rotate_on_resample**: Auto-rotate landscape images (default: true)
- **apply_dewarping**: Apply dewarping for curved paper (default: false)

### Data Paths

- **test_csv_path**: Path to competition test.csv
- **test_images_dir**: Directory containing test images
- **submission_path**: Output path for submission.csv

### Layout Configuration

The script uses the George Moody 2024 layout templates by default:
- `src/config/lead_layouts_george-moody-2024.yml`

This includes various ECG layouts commonly found in clinical practice. If your test images have different layouts, you can switch to:
- `src/config/lead_layouts_all.yml` - Comprehensive layout library
- `src/config/lead_layouts_reduced.yml` - Minimal layout set

## Pipeline Details

### Processing Steps

1. **Load Image**: Decode PNG image and prepare for inference
2. **Semantic Segmentation**: U-Net identifies ECG traces, grids, and background
3. **Perspective Correction**: Corrects image distortions and rotations
4. **Grid Calibration**: Estimates pixel-to-millimeter scaling from grid
5. **Signal Extraction**: Converts segmented traces to time series
6. **Lead Identification**: Matches extracted signals to 12-lead layout
7. **Canonicalization**: Reorders to standard lead order [I, II, III, aVR, aVL, aVF, V1-V6]
8. **Resampling**: Adjusts signal length based on sampling frequency
   - Lead II: `floor(fs * 10)` samples (10 seconds)
   - Others: `floor(fs * 2.5)` samples (2.5 seconds)
9. **Unit Conversion**: Converts from µV to mV (divide by 1000)
10. **Format Conversion**: Transforms to competition submission format

### Key Adaptations

#### Duration Handling

The model outputs fixed-length signals (default 5000 samples). The integration script resamples each lead to the correct duration:

```python
# Lead II: 10 seconds
lead_ii_samples = int(np.floor(fs * 10))

# Other leads: 2.5 seconds
other_samples = int(np.floor(fs * 2.5))
```

#### Unit Conversion

The model outputs signals in microvolts (µV). The competition requires millivolts (mV):

```python
signal_mv = signal_uv / 1000.0
```

#### Lead Order

Standard 12-lead order:
```
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

## Troubleshooting

### Image Not Found

If you see warnings like `Warning: Image not found: /path/to/{id}.png`, ensure:
1. The `test_images_dir` path is correct
2. Image files have `.png` extension
3. Image filenames match IDs in test.csv

### No Signals Extracted

If you see `Warning: No signals extracted for {id}`, the model failed to digitize the ECG. This can happen with:
- Extremely poor image quality
- Unusual ECG layouts not in the template library
- Severely distorted or damaged images

The script will output NaN values for these cases.

### GPU Memory Issues

If you encounter CUDA out-of-memory errors:
1. Reduce `resample_size` in the config (e.g., 2000 or 1500)
2. Switch to CPU processing (slower but no memory limit)
3. Process images in smaller batches

### Layout Matching Failures

If the layout identifier fails to match correctly:
1. Check if images contain all 12 leads or a subset
2. Try using `lead_layouts_all.yml` for more layout options
3. Enable debug mode: `MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.debug=true`

## Performance Optimization

### Speed Improvements

1. **Use GPU**: CUDA-enabled GPU provides 10-50x speedup
2. **Reduce resample_size**: Smaller images process faster (may reduce accuracy)
3. **Disable timing**: Set `enable_timing=false` (default)

### Accuracy Improvements

1. **Enable dewarping**: For curved/folded paper, set `apply_dewarping=true`
2. **Use comprehensive layouts**: Switch to `lead_layouts_all.yml`
3. **Enable flip detection**: Set `possibly_flipped=true` for upside-down images

## Expected Output

After successful execution, you'll see:

```
Loading test metadata from /kaggle/input/physionet-ecg-image-digitization/test.csv
Found 1000 unique images to process
Initializing inference model...
Model initialized successfully
Processing images: 100%|███████████████| 1000/1000 [XX:XX<00:00,  X.XXit/s]
Creating submission file...
Submission saved to ./submission.csv
Total predictions: 60000
```

The submission file will contain approximately 60,000 rows:
- ~1000 images × 12 leads × ~5 samples per lead (varies by fs)

## Competition Submission

1. Ensure `submission.csv` has the correct format:
   ```
   id,value
   '62_0_I',0.0
   '62_1_II',0.3
   ...
   ```

2. Submit to Kaggle:
   - Via web interface: Upload `submission.csv`
   - Via API: `kaggle competitions submit -c physionet-ecg-image-digitization -f submission.csv -m "Message"`

## Model Performance

The Open-ECG-Digitizer model was designed for:
- Clinical ECG digitization
- Robust handling of image artifacts
- Multiple lead layout configurations

Expected performance on the competition:
- High accuracy on clean scanned images
- Good robustness to perspective distortions
- Handles various ECG layouts and configurations

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{stenhede_digitizing_2025,
  title        = {Digitizing Paper {ECGs} at Scale: An Open-Source Algorithm for Clinical Research},
  author       = {Stenhede, Elias and Bjørnstad, Agnar Martin and Ranjbar, Arian},
  year         = {2025},
  doi          = {10.48550/ARXIV.2510.19590},
  shorttitle   = {Digitizing Paper {ECGs} at Scale}
}
```

## Additional Resources

- **Competition Page**: https://www.kaggle.com/competitions/physionet-ecg-image-digitization
- **Model Repository**: https://github.com/Ahus-AIM/Electrocardiogram-Digitization
- **Paper**: https://arxiv.org/abs/2510.19590

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: elias.stenhede at ahus.no
