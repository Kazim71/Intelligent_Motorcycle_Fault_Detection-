# Intelligent Motorcycle Fault Detection Using ML-Based Acoustic Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Audio Processing](https://img.shields.io/badge/Audio-Librosa-green.svg)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-1%20Complete-brightgreen.svg)](PHASE1_SETUP.md)

## 🚀 Phase 1 - Data Pipeline Rebuild Complete!

**Latest Update**: Phase 1 comprehensive rebuild with advanced EDA, feature extraction, and data augmentation pipeline.

> **🎯 Quick Start**: See [PHASE1_SETUP.md](PHASE1_SETUP.md) for detailed setup instructions and usage guide.

## 🎯 Project Overview

This project introduces an intelligent fault detection system for motorcycles that leverages machine learning techniques on engine acoustic signals, enabling non-intrusive and efficient diagnosis. The system captures audio signals from a running motorcycle and processes them using advanced audio feature extraction techniques to accurately classify engine conditions and detect potential faults.

## 🆕 Phase 1 Features (Recently Implemented)

### 1. 📊 Comprehensive Exploratory Data Analysis
- **Class Distribution Analysis**: Visual and statistical analysis of fault categories
- **Audio Visualization**: Waveforms and mel-spectrograms for each fault type
- **Statistical Profiling**: Duration, amplitude, energy analysis per category
- **Imbalance Detection**: Automatic identification of dataset imbalances

### 2. 🎵 Advanced Feature Extraction Pipeline
- **MFCC Features**: 20 coefficients with statistical measures (mean, variance, skewness, kurtosis)
- **Mel-Spectrogram**: 128 mel bands with comprehensive statistics
- **Spectral Features**: Centroid, bandwidth, rolloff, contrast, flatness with temporal statistics
- **Chromagram**: 12 chroma bins for harmonic content analysis
- **Temporal Features**: Zero-crossing rate, RMS energy, tempo, beat tracking
- **Tonnetz Features**: Tonal centroid representation (6 dimensions)
- **Advanced Features**: Signal energy, spectral entropy, harmonic/percussive ratios
- **Total**: 600+ features per audio file

### 3. 🔄 Data Augmentation & Noise Handling
- **Gaussian Noise Addition**: Variable SNR levels (10-30 dB)
- **Time Stretching**: Speed variations (0.8x - 1.2x) without pitch change
- **Pitch Shifting**: ±2 semitones for tonal variations
- **Room Impulse Response**: Simulates different acoustic environments
- **Dynamic Range Compression**: Professional audio processing
- **Advanced Noise Reduction**: Using spectral subtraction and noisereduce library
- **Dataset Expansion**: 4x augmentation factor for improved generalization


### Team Members
- **Mohammad Kazim** (PRN: 2114110466) - Project Lead & System Integration
- **Aryan Gupta** (PRN: 2114110452) - Model Training & UI Development  
- **Faiz Maqsood** (PRN: 2114110489) - Data Collection Specialist

### Mentor
- **Dr. Harshada Mhetre** - Guide, ECE Department

## 🔧 Key Features

- **Real-Time Fault Detection**: Identifies engine-related issues within 1.5 seconds
- **High Accuracy**: Achieves 94.7% classification accuracy using Random Forest
- **Non-Intrusive Monitoring**: Uses only acoustic data without hardware modifications
- **Multi-Fault Classification**: Detects brake, chain, engine, and silencer faults
- **Cost-Effective**: Requires only a microphone and basic computing device
- **User-Friendly Interface**: Simple GUI for testing and diagnostics

## 🏗️ System Architecture

### Data Flow
1. **Audio Input Module**: Captures engine sound using microphone
2. **Signal Preprocessing**: Applies denoising and normalization
3. **Feature Extraction**: Uses MFCCs, ZCR, Spectral Centroid
4. **ML Classification**: Employs Random Forest, SVM, and KNN algorithms
5. **Result Interface**: Displays fault types and confidence scores

### Block Diagram
```
[Audio Input] → [Preprocessing] → [Feature Extraction] → [ML Models] → [Classification Results]
```

## 🤖 Machine Learning Models

### Model Performance
| Model | Accuracy | F1-Score | Best Use Case |
|-------|----------|----------|---------------|
| **Random Forest** | **94.7%** | 0.94 | Overall best performance |
| SVM | 84.2% | 0.85 | Robust to noise |
| KNN | 84.2% | 0.85 | Simple implementation |

### Fault Detection Categories
- **Engine Faults** (68 samples)
- **Brake Faults** (49 samples) 
- **Chain Issues** (52 samples)
- **Silencer Issues** (49 samples)

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Librosa** | Audio signal processing and feature extraction |
| **Scikit-learn** | Machine learning model implementation |
| **Matplotlib & Seaborn** | Data visualization |
| **NumPy & Pandas** | Data manipulation and analysis |
| **Audacity** | Audio recording and cleaning |

## 📊 Technical Implementation

### Feature Extraction
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Primary audio features
- **Chroma Features**: Harmonic content analysis
- **Spectral Contrast**: Frequency band energy differences
- **Zero Crossing Rate**: Time-domain characteristics
- **RMS Energy**: Signal amplitude analysis

### Data Processing Pipeline
```python
# Feature extraction workflow
Audio Signal → Noise Filtering → MFCC Extraction → Model Training → Fault Classification
```

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
Audio dataset (.wav files)
```

### Phase 1 Quick Setup
```bash
# Clone the repository
git clone https://github.com/Kazim71/Intelligent_Motorcycle_Fault_Detection-.git
cd Intelligent_Motorcycle_Fault_Detection-

# Install dependencies
pip install -r requirements.txt

# Optional: Advanced noise reduction
pip install noisereduce

# Run Phase 1 Pipeline
python run_phase1.py --dataset_path ./data/raw/
```

### Project Structure (Phase 1)
```
├── src/                           # Source code
│   ├── 01_exploratory_data_analysis.py
│   ├── 02_advanced_feature_extraction.py
│   └── 03_data_augmentation_noise_handling.py
├── data/                          # Data directories
│   ├── raw/                       # Original audio files
│   ├── processed/                 # Extracted features
│   └── augmented/                 # Augmented dataset  
├── eda/                          # EDA outputs
├── models/                       # Trained models (Phase 2)
├── run_phase1.py                 # Main Phase 1 pipeline
├── requirements.txt              # Python dependencies
└── PHASE1_SETUP.md              # Detailed setup guide
```

## 📈 Results & Performance

### Classification Results
- **Random Forest**: 94.7% accuracy with perfect recall for brake faults
- **Real-time Processing**: Average prediction time of 1.3 seconds
- **Component-specific Detection**: High precision for engine and silencer faults
- **Minimal Misclassification**: Strong diagonal values in confusion matrix

### Key Achievements
- ✅ Successfully detects 4 critical fault categories
- ✅ Real-time processing capability
- ✅ High accuracy across different motorcycle models
- ✅ User-friendly interface for practical deployment

## 🎯 Applications

### Target Users
- **Mechanics & Workshops**: Replace guesswork with sound-based diagnostics
- **Individual Motorcycle Owners**: DIY fault detection
- **Service Centers**: Automated diagnostic tools
- **Fleet Management**: Remote monitoring capabilities

### Use Cases
- Preventive maintenance scheduling
- Emergency roadside diagnostics
- Workshop efficiency improvement
- Training tool for new mechanics

## 🔬 Research Contributions

### Novel Aspects
1. **Multi-fault Classification**: Simultaneously detects multiple fault types
2. **Real-time Processing**: Practical deployment capability
3. **Cost-effective Solution**: Uses standard hardware
4. **High Accuracy**: Outperforms traditional diagnostic methods

### Academic Impact
- Demonstrates feasibility of acoustic-based vehicle diagnostics
- Provides benchmark for ML-based fault detection systems
- Contributes to predictive maintenance research

## 🚧 Challenges & Limitations

### Current Limitations
- Background noise interference in outdoor environments
- Performance variation across different motorcycle brands
- Data imbalance for certain fault types
- Slight latency in browser/mobile deployments

### Proposed Solutions
- Advanced noise filtering algorithms
- Expanded training dataset
- Data augmentation techniques
- Hardware optimization

## 🔮 Future Scope

### Short-term Enhancements
- [ ] Mobile app development
- [ ] Real-time streaming capability
- [ ] Expanded fault categories
- [ ] Multi-language support

### Long-term Vision
- [ ] Integration with IoT devices
- [ ] Cloud-based analytics
- [ ] Deep learning models (CNN/RNN)
- [ ] Extension to other vehicle types
- [ ] Fleet management integration

## 📚 Documentation

### Project Structure
```
├── README.md
├── dataset/                 # Audio samples
├── models/                 # Trained ML models
├── results/                # Performance metrics
├── src/                    # Source code
└── docs/                   # Documentation
```

### Key Files
- **Classification Models**: Random Forest, SVM, KNN implementations
- **Feature Extraction**: MFCC and spectral analysis scripts
- **Performance Evaluation**: Accuracy metrics and confusion matrices
- **Visualization**: Results plotting and analysis

## 🤝 Contributing

We welcome contributions to improve the system:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dr. Harshada Mhetre** for invaluable guidance and mentorship
- **Prof. (Dr.) Arundhati A. Shinde** for departmental support
- **Bharati Vidyapeeth College of Engineering** for providing research opportunity
- All participants who contributed motorcycle audio samples for training

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Mohammad Kazim**: [GitHub Profile](https://github.com/Kazim71)
- **Project Repository**: [Intelligent_Motorcycle_Fault_Detection](https://github.com/Kazim71/Intelligent_Motorcycle_Fault_Detection-)

---

**Note**: This project represents a significant step toward integrating AI-based diagnostics into two-wheeler maintenance ecosystems and demonstrates the practical application of machine learning in automotive engineering.
