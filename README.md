# Intelligent Motorcycle Fault Detection Using ML-Based Acoustic Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Audio Processing](https://img.shields.io/badge/Audio-Librosa-green.svg)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This project introduces an intelligent fault detection system for motorcycles that leverages machine learning techniques on engine acoustic signals, enabling non-intrusive and efficient diagnosis. The system captures audio signals from a running motorcycle and processes them using advanced audio feature extraction techniques to accurately classify engine conditions and detect potential faults.

## 🏆 Project Team

**Academic Year: 2024-2025**  
**Department of Electronics and Communication Engineering**  
**Bharati Vidyapeeth (Deemed to be University) College of Engineering, Pune**

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
Microphone for audio input
```

### Dependencies
```bash
pip install numpy pandas librosa scikit-learn matplotlib seaborn
```

### Quick Start
1. Clone the repository
2. Install dependencies
3. Run the main classification script
4. Input motorcycle audio for analysis

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

*Developed with ❤️ by the ECE Department, Bharati Vidyapeeth College of Engineering, Pune*