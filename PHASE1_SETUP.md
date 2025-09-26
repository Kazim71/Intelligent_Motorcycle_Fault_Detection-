# Phase 1 Setup Guide - Intelligent Motorcycle Fault Detection

## Quick Start Guide

### 1. Installation
```bash
# Clone the repository (if not already done)
git clone https://github.com/Kazim71/Intelligent_Motorcycle_Fault_Detection-.git
cd Intelligent_Motorcycle_Fault_Detection-

# Install required packages
pip install -r requirements.txt

# Optional: Install additional packages for better noise reduction
pip install noisereduce
```

### 2. Dataset Preparation
```bash
# Create data directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/augmented

# Place your .wav audio files in data/raw/
# Organize files with labels in filenames or folder structure:
# Example: engine_fault_001.wav, brake_issue_002.wav, etc.
```

### 3. Run Phase 1 Pipeline
```bash
# Run complete Phase 1 pipeline
python run_phase1.py --dataset_path ./data/raw/

# Check dependencies only
python run_phase1.py --check_deps_only

# Skip EDA (if already done)
python run_phase1.py --dataset_path ./data/raw/ --skip_eda

# Skip augmentation (for faster processing)
python run_phase1.py --dataset_path ./data/raw/ --skip_augmentation
```

### 4. Individual Script Execution
```bash
# Run only EDA
python src/01_exploratory_data_analysis.py

# Run only feature extraction  
python src/02_advanced_feature_extraction.py

# Run only data augmentation
python src/03_data_augmentation_noise_handling.py
```

## Phase 1 Components

### 1. Exploratory Data Analysis (EDA)
**File**: `src/01_exploratory_data_analysis.py`

**Features**:
- ✅ Class distribution analysis with visualizations
- ✅ Random sample visualization (waveforms + mel-spectrograms)
- ✅ Audio statistics calculation (duration, amplitude, energy, etc.)
- ✅ Statistical analysis by category
- ✅ Class imbalance detection
- ✅ Comprehensive EDA report generation

**Outputs**:
- `./eda/class_distribution.png`
- `./eda/[category]_samples_visualization.png`
- `./eda/audio_statistics.csv`
- `./eda/audio_statistics_by_category.png`
- `./eda/eda_report.md`

### 2. Advanced Feature Extraction
**File**: `src/02_advanced_feature_extraction.py`

**Features**:
- ✅ **MFCC Features**: 20 coefficients with mean, variance, skewness, kurtosis
- ✅ **Mel-Spectrogram**: 128 mel bands with statistical measures
- ✅ **Spectral Features**: Centroid, bandwidth, rolloff, contrast, flatness
- ✅ **Chromagram**: 12 chroma bins with statistics
- ✅ **Temporal Features**: Zero-crossing rate, RMS energy, tempo, beats
- ✅ **Tonnetz**: Tonal centroid features (6 dimensions)
- ✅ **Additional Features**: Signal energy, spectral entropy, harmonic/percussive ratios

**Total Features**: 600+ comprehensive audio features per file

**Outputs**:
- `./data/processed/extracted_features.csv`
- `./data/processed/features.npy`
- `./data/processed/labels.npy`
- `./data/processed/feature_names.txt`
- `./data/processed/feature_extractor.pkl`

### 3. Data Augmentation & Noise Handling
**File**: `src/03_data_augmentation_noise_handling.py`

**Augmentation Techniques**:
- ✅ **Gaussian Noise Addition**: Various SNR levels (10-30 dB)
- ✅ **Time Stretching**: Speed variation (0.8x - 1.2x)
- ✅ **Pitch Shifting**: ±2 semitones
- ✅ **Room Impulse Response**: Small room, large hall, garage simulation
- ✅ **Dynamic Range Compression**: Threshold-based compression

**Noise Reduction**:
- ✅ **Advanced Noise Reduction**: Using noisereduce library (if available)
- ✅ **Spectral Subtraction**: Fallback noise reduction method
- ✅ **Background Noise Handling**: Automatic noise estimation and removal

**Outputs**:
- `./data/processed/augmented/original_clean/`: Noise-reduced originals
- `./data/processed/augmented/augmented/`: 4x augmented versions per file

## Expected Workflow

### Step 1: Exploratory Data Analysis
```python
from src.01_exploratory_data_analysis import MotorcycleAudioEDA

eda = MotorcycleAudioEDA(dataset_path="./data/raw/")
results = eda.run_complete_eda()
```

### Step 2: Feature Extraction
```python
from src.02_advanced_feature_extraction import AdvancedFeatureExtractor

extractor = AdvancedFeatureExtractor(n_mfcc=20, n_mels=128)
feature_df = extractor.process_dataset("./data/raw/", "./data/processed/")
```

### Step 3: Data Augmentation
```python
from src.03_data_augmentation_noise_handling import DataAugmentationPipeline

augmenter = DataAugmentationPipeline(augmentation_factor=4)
augmenter.process_dataset("./data/raw/", "./data/processed/augmented/")
```

## Configuration Options

### AdvancedFeatureExtractor Parameters
```python
extractor = AdvancedFeatureExtractor(
    sample_rate=22050,    # Target sample rate
    n_mfcc=20,           # Number of MFCC coefficients
    n_mels=128,          # Number of mel bands
    hop_length=512       # Hop length for STFT
)
```

### DataAugmentationPipeline Parameters
```python
augmenter = DataAugmentationPipeline(
    sample_rate=22050,        # Target sample rate
    augmentation_factor=4     # Number of augmented versions per file
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **No Audio Files Found**
   - Check dataset path
   - Ensure files have .wav extension
   - Verify file permissions

3. **Memory Issues with Large Datasets**
   - Process files in batches
   - Reduce n_mels parameter
   - Use shorter audio duration

4. **Noise Reduction Issues**
   ```bash
   pip install noisereduce
   ```

### Performance Optimization

1. **For Large Datasets**:
   - Use multiprocessing (implement in custom version)
   - Process files in chunks
   - Use SSD storage for faster I/O

2. **For Limited Memory**:
   - Reduce feature dimensions
   - Process shorter audio segments
   - Use data streaming

## Validation Checklist

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset path exists and contains .wav files
- [ ] EDA generates plots and statistics
- [ ] Feature extraction produces feature matrix
- [ ] Data augmentation creates augmented files
- [ ] All output directories created successfully
- [ ] Feature names and labels saved correctly

## Phase 1 Success Criteria

✅ **EDA Complete**: Class distribution analyzed, audio samples visualized  
✅ **Features Extracted**: 600+ features per audio file  
✅ **Data Augmented**: 4x dataset size with noise reduction  
✅ **Files Generated**: All CSV, NPY, and visualization files created  
✅ **Pipeline Functional**: Can run end-to-end without errors  

## Next Phase Preview

**Phase 2 will include**:
- Model training with extracted features
- Hyperparameter optimization
- Cross-validation setup
- Performance evaluation
- Model comparison (Random Forest, SVM, KNN, Deep Learning)
- Real-time inference pipeline

---

*Phase 1 Implementation by Mohammad Kazim, Aryan Gupta, Faiz Maqsood*  
*Bharati Vidyapeeth College of Engineering, Pune*