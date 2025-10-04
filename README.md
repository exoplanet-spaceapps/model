# NASA Kepler Exoplanet Classification Project

A machine learning project for classifying Kepler Objects of Interest (KOI) using multiple approaches including deep learning (GP+CNN pipeline), traditional ML (Random Forest, XGBoost), and neural networks.

## 🎯 Project Overview

This project aims to identify potential exoplanets from NASA's Kepler mission data by analyzing light curves and extracted features. The project implements a complete ML pipeline from Gaussian Process denoising to deep learning classification.

### Key Features

- **GP Denoising**: Gaussian Process regression for light curve preprocessing
- **TLS Search**: Transit Least Squares for period detection
- **Deep Learning**: CNN-based classification pipeline (GP+CNN)
- **Traditional ML**: Random Forest, XGBoost with GPU acceleration
- **Neural Networks**: Multiple architectures (MLP, 1D-CNN, GP+CNN)
- **Comprehensive Benchmarking**: CPU vs GPU performance comparison

## 📊 Dataset

### Input Data Files (in `data/`)

1. **`tsfresh_features.csv`** (21.6 MB)
   - Time-series features extracted using TSFresh library
   - Contains ~3,500 samples with extracted statistical features
   - Used for traditional ML models (Random Forest, XGBoost, MLP)

2. **`q1_q17_dr25_koi.csv`**
   - Kepler Objects of Interest catalog (Quarters 1-17, Data Release 25)
   - Contains KOI metadata: period, t0, duration, disposition
   - Fields: `kepid`, `kepoi_name`, `kepler_name`, `koi_disposition`, `koi_pdisposition`

### Data Preprocessing

- Remove columns with single unique values
- Filter out samples with infinity values
- Fill NaN values with zero or column mean
- StandardScaler normalization
- Train/Val/Test split: varies by model (typically 90%/5%/5%)

## 🏗️ Project Structure

```
model/
├── app/                          # Application modules
│   ├── models/
│   │   └── cnn1d.py             # Two-Branch 1D-CNN implementation
│   ├── data/
│   │   └── fold.py              # Phase folding & view construction
│   ├── trainers/
│   │   ├── cnn1d_trainer.py     # Training loop for CNN
│   │   └── utils.py             # Utility functions
│   ├── calibration/
│   │   └── calibrate.py         # Model calibration utilities
│   ├── denoise/                 # GP denoising modules
│   ├── search/                  # TLS period search
│   └── validation/              # Validation utilities
├── notebooks/                    # Jupyter notebooks
│   ├── 03b_cnn_train.ipynb      # CNN training
│   └── 04_newdata_inference.ipynb # Inference pipeline
├── scripts/                      # Executable scripts
│   ├── benchmarks/              # Performance benchmarking
│   │   ├── complete_gpcnn_benchmark.py    # Complete GP+CNN benchmark
│   │   ├── ultraoptimized_benchmark.py    # Ultra-optimized comparison
│   │   ├── ultraoptimized_cpu_models.py   # CPU-optimized models
│   │   ├── ultraoptimized_gpu_models.py   # GPU-optimized models
│   │   └── visualize_gpcnn_comparison.py  # Visualization tools
│   └── legacy/                  # Legacy training scripts
│       ├── koi_project_nn.py    # Simple neural network (MLP)
│       ├── train_rf_v1.py       # Random Forest classifier
│       └── xgboost_koi.py       # XGBoost classifier
├── data/                         # Data files (gitignored if large)
│   ├── tsfresh_features.csv     # Extracted features
│   └── q1_q17_dr25_koi.csv      # KOI catalog
├── reports/                      # Generated reports & results
│   ├── figures/                 # Plots and visualizations (PDF/PNG)
│   ├── results/                 # Model results (JSON)
│   ├── FINAL_GPU_BENCHMARK_REPORT.txt
│   ├── FINAL_GPU_RESULTS_REPORT.md
│   ├── GP_CNN_COMPLETE_ANALYSIS.md
│   └── ULTRA_OPTIMIZATION_FINAL_REPORT.md
├── SPECS/                        # Technical specifications
│   ├── 1D_CNN_SPEC.md           # CNN architecture spec
│   ├── PIPELINE_SPEC.md         # Full pipeline specification
│   └── INTEGRATION_PLAN.md      # Integration guidelines
├── prompts/                      # Development workflow prompts
│   └── claude-commands.md       # Claude Code automation
├── docs/                         # Documentation
├── patches/                      # Code patches for upgrades
├── .claude/                      # Claude Code configuration
├── CLAUDE.md                     # Development guide
├── CITATIONS.md                  # References
├── README_UPGRADE.md             # Upgrade instructions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` - Deep learning framework
- `numpy`, `pandas` - Data processing
- `scikit-learn` - Traditional ML
- `xgboost` - Gradient boosting
- `transitleastsquares` - Period detection
- `celerite2` or `starry_process` - GP denoising (optional)

### 2. Running Benchmarks

#### Complete GP+CNN Benchmark
```bash
python scripts/benchmarks/complete_gpcnn_benchmark.py
```
Runs comprehensive benchmark including:
- GP+CNN pipeline
- Neural Networks (Simple MLP, Heavy NN)
- XGBoost (GPU & CPU)
- Random Forest (CPU)

#### Ultra-Optimized Models
```bash
python scripts/benchmarks/ultraoptimized_benchmark.py
```
Tests 2025 best practices:
- GPU optimizations (cuDNN, TF32, Mixed Precision)
- CPU optimizations (Intel MKL, OpenMP)
- Comparison across all model types

#### Visualization
```bash
python scripts/benchmarks/visualize_gpcnn_comparison.py
```
Generates comparison plots and reports.

### 3. Legacy Models

#### Simple Neural Network (MLP)
```bash
python scripts/legacy/koi_project_nn.py
```
- 3-layer MLP: 256→64→1
- BatchNorm + GELU activation
- AdamW optimizer (lr=3e-5)

#### Random Forest
```bash
python scripts/legacy/train_rf_v1.py
```
- Optimized: depth=8, n_estimators=200
- Grid search with cross-validation

#### XGBoost
```bash
python scripts/legacy/xgboost_koi.py
```
- Gradient boosting with tree-based learning

## 🎯 Model Architectures

### 1. GP+CNN Pipeline (Recommended)

**Pipeline:**
1. **GP Denoising**: Remove systematics using Gaussian Process regression
2. **TLS Search**: Detect periods with Transit Least Squares
3. **CNN Classification**: Deep learning on denoised light curves

**Architecture:**
- GP simulator: Linear(input) → 1024 → 2048
- CNN layers: Conv1D blocks with BatchNorm
- Classifier: FC layers with LayerNorm and GELU
- Optimizations: Mixed precision (AMP), GPU acceleration

**Key Features:**
- Handles raw light curves (no manual feature engineering)
- Multi-scale pattern recognition
- GPU-optimized for fast training
- Best for transit morphology analysis

### 2. Traditional ML Models

**Random Forest:**
- Best for TSFresh features
- No GPU required
- Excellent interpretability
- Achieves ~88% ROC-AUC

**XGBoost:**
- GPU acceleration available
- Fast training on large datasets
- Good balance of speed and accuracy
- Achieves ~87% ROC-AUC

### 3. Neural Networks

**Simple MLP:**
- Lightweight baseline
- Fast training
- Good for feature-based data

**Heavy NN:**
- 5-layer deep network
- GPU-optimized
- Mixed precision training

## 📈 Performance Benchmarks

### Latest Results (from reports/)

| Model | ROC-AUC | Accuracy | Device | Training Time | GPU Util |
|-------|---------|----------|--------|---------------|----------|
| Random Forest | 0.881 | 81.6% | CPU | 14.2s | N/A |
| XGBoost (GPU) | 0.871 | 79.9% | GPU | 3.4s | 84% |
| GP+CNN | 0.734 | 62.9% | GPU | 25.8s | 100% |
| Heavy NN | 0.683 | 65.0% | GPU | 8.9s | 7% |
| Simple MLP | 0.667 | 61.8% | GPU | 2.1s | 0% |

**Key Findings:**
- Random Forest (CPU) achieves best performance on TSFresh features
- XGBoost shows excellent GPU utilization (84%) with 4x speedup
- GP+CNN designed for raw light curves; underperforms on extracted features
- Tree-based models excel on tabular data

### GPU Optimizations Applied

**GPU Models:**
- ✅ cuDNN benchmark mode
- ✅ TF32 for Tensor Cores
- ✅ Mixed Precision (AMP)
- ✅ Pinned memory transfers
- ✅ Non-blocking data loading
- ✅ Batch size tuning (divisible by 8)

**CPU Models:**
- ✅ Physical core allocation
- ✅ MKL/OpenMP threading
- ✅ Memory-aligned arrays
- ✅ Intel Extension (if available)

## 🔧 Development Workflow

### Using Claude Code

This project includes automation for development with Claude:

1. **Setup**: Read `CLAUDE.md` for guidelines
2. **Specs**: Review detailed specifications in `SPECS/`
3. **Prompts**: Execute workflows from `prompts/claude-commands.md`
4. **Benchmarks**: Run scripts in `scripts/benchmarks/` for performance testing

### Adding New Models

1. **Implement** in `app/models/`
2. **Add trainer** in `app/trainers/`
3. **Create benchmark** in `scripts/benchmarks/`
4. **Document** performance in `reports/`

## 📊 Results & Reports

All benchmark results and reports are stored in `reports/`:

- **Figures**: `reports/figures/*.{png,pdf}` - Visualizations
- **Results**: `reports/results/*.json` - Numerical results
- **Reports**: `reports/*.md` - Analysis and findings

## 🎓 Technical Background

### Transit Method
Detect periodic dips in stellar brightness when an orbiting planet passes in front of the star.

### Kepler Mission
NASA space telescope that monitored 150,000+ stars (2009-2018).

### KOI Classification
Distinguish true planetary transits from false positives (eclipsing binaries, stellar variability).

## 📝 Citation

If you use this code, please cite:
- Kepler Mission: https://www.nasa.gov/kepler
- NASA Exoplanet Archive
- See `CITATIONS.md` for detailed references

## 📄 License

This project analyzes public NASA Kepler mission data.

## 🤝 Contributing

1. Review specifications in `SPECS/`
2. Follow coding patterns in `app/`
3. Add tests for new features
4. Update documentation and benchmarks

---

**Note**: This project demonstrates a complete ML pipeline for exoplanet detection, from GP denoising to deep learning classification. The GP+CNN pipeline represents the modern approach for raw light curve analysis, while feature-based models (RF, XGBoost) provide strong baseline performance on extracted features.
