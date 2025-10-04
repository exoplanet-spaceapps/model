# NASA Kepler Exoplanet Classification Project

A machine learning project for classifying Kepler Objects of Interest (KOI) using multiple approaches including Neural Networks, Random Forest, XGBoost, and a Two-Branch 1D-CNN architecture optimized for transit light curve analysis.

## 🎯 Project Overview

This project aims to identify potential exoplanets from NASA's Kepler mission data by analyzing light curves and extracted features. The project implements multiple classification models:

- **Simple Neural Network** (MLP): 3-layer feedforward network with BatchNorm and GELU activation
- **Random Forest**: Ensemble classifier with optimized hyperparameters
- **XGBoost**: Gradient boosting classifier
- **Two-Branch 1D-CNN** ⭐: State-of-the-art dual-branch convolutional architecture for global and local view analysis

## 📊 Dataset

### Input Data Files

1. **`tsfresh_features.csv`** (21.6 MB)
   - Time-series features extracted using TSFresh library
   - Contains ~3,500 samples with extracted statistical features
   - Used for traditional ML models (Random Forest, XGBoost, Simple NN)

2. **`q1_q17_dr25_koi.csv`**
   - Kepler Objects of Interest catalog (Quarters 1-17, Data Release 25)
   - Contains KOI metadata: period, t0, duration, disposition
   - Fields: `kepid`, `kepoi_name`, `kepler_name`, `koi_disposition`, `koi_pdisposition`

### Data Preprocessing

- Remove columns with single unique values
- Filter out samples with infinity values
- Fill NaN values with zero or column mean
- MinMax scaling to [0, 1] range
- Train/Val/Test split: varies by model (typically 90%/5%/5%)

## 🏗️ Architecture

### Project Structure

```
model/
├── app/
│   ├── models/
│   │   └── cnn1d.py              # Two-Branch 1D-CNN implementation
│   ├── data/
│   │   └── fold.py               # Phase folding & view construction
│   ├── trainers/
│   │   ├── cnn1d_trainer.py      # Training loop for CNN
│   │   └── utils.py              # Utility functions
│   └── calibration/
│       └── calibrate.py          # Model calibration utilities
├── notebooks/
│   ├── 03b_cnn_train_mps.ipynb   # CNN training on MPS (M-series Mac)
│   └── 04_newdata_inference.ipynb # Inference on new data
├── SPECS/
│   ├── 1D_CNN_SPEC.md            # Detailed CNN specifications
│   ├── INTEGRATION_PLAN.md        # Integration guidelines
│   └── model_card_template.md     # Model documentation template
├── prompts/
│   └── claude-commands.md         # Development workflow prompts
├── patches/                       # Code patches for upgrades
├── koi_project_nn.py             # Simple neural network (MLP)
├── train_rf_v1.py                # Random Forest classifier
├── xgboost_koi.py                # XGBoost classifier
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Development guide
└── README.md                     # This file
```

### Two-Branch 1D-CNN Architecture

The modern CNN approach uses a dual-branch architecture:

**Global Branch** (app/models/cnn1d.py:40)
- Input: Full phase-folded light curve (2000 time steps)
- Conv layers with kernels: [7, 5, 5]
- Captures overall transit morphology

**Local Branch** (app/models/cnn1d.py:41)
- Input: Zoomed transit region (512 time steps)
- Conv layers with kernels: [5, 3, 3]
- Captures fine-grained transit features

**Architecture Details:**
```
ConvBlock: Conv1D → BatchNorm → ReLU → MaxPool
Branch: ConvBlock × 3 → Global Average Pooling
TwoBranchCNN1D:
  ├── Global Branch (32 channels → 64 channels)
  ├── Local Branch (32 channels → 64 channels)
  ├── Concatenate [128 features]
  ├── FC1 (128) → ReLU → Dropout(0.3)
  └── FC2 (1) → Logits
```

**Input Processing** (app/data/fold.py)
- Phase folding with period and t0
- Robust normalization using MAD (Median Absolute Deviation)
- Equal-spaced resampling for fixed-length input
- Global view: entire phase [0, 1]
- Local view: transit window (k×duration/period)

## 🚀 Usage

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- numpy <= 2.0
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch (for CNN models)

### Running Models

#### 1. Simple Neural Network (MLP)
```bash
python koi_project_nn.py
```
- 3-layer MLP with 256→64→1 architecture
- BatchNorm + GELU activation
- AdamW optimizer (lr=3e-5)
- Early stopping (patience=30)

#### 2. Random Forest
```bash
python train_rf_v1.py
```
- Optimized hyperparameters: depth=8, n_estimators=200
- Grid search with cross-validation
- Feature importance analysis

#### 3. XGBoost
```bash
python xgboost_koi.py
```
- Gradient boosting with tree-based learning

#### 4. Two-Branch 1D-CNN (Recommended)

**Training:**
```python
from app.models.cnn1d import make_model
from app.trainers.cnn1d_trainer import train
from app.data.fold import LightCurveViewsDataset, Item

# Create dataset
items = [Item(time, flux, period, t0, duration, label), ...]
train_ds = LightCurveViewsDataset(items, g_len=2000, l_len=512)

# Train model
model = make_model()
metrics = train(
    model, train_ds, val_ds,
    device="mps",  # or "cuda" or "cpu"
    batch_size=64,
    lr=1e-3,
    max_epochs=50,
    patience=7,
    workdir="./outputs"
)
```

**Jupyter Notebooks:**
- `notebooks/03b_cnn_train_mps.ipynb`: Training on M-series Mac (MPS backend)
- `notebooks/04_newdata_inference.ipynb`: Inference pipeline

### Device Support

The CNN implementation supports multiple backends:
- **MPS**: Apple Silicon (M1/M2/M3/M4 Mac)
- **CUDA**: NVIDIA GPUs
- **CPU**: Fallback for all platforms

Device selection:
```python
device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
```

## 📈 Training Details

### CNN Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | AdamW | Weight decay regularization |
| Learning Rate | 1e-3 | Initial learning rate |
| Weight Decay | 1e-4 | L2 regularization |
| Scheduler | ReduceLROnPlateau | Adaptive LR (factor=0.5, patience=2) |
| Loss Function | BCEWithLogitsLoss | Binary cross-entropy with logits |
| Batch Size | 64 | Training batch size |
| Max Epochs | 50 | Maximum training epochs |
| Early Stopping | Patience=7 | Stop if no improvement in val AP |

### Evaluation Metrics

- **Primary**: Average Precision (AP) / PR-AUC
- **Secondary**: ROC-AUC
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion Matrix, Loss curves

### Output Files

After training, the following files are generated:

```
workdir/
├── artifacts/
│   └── cnn1d.pt                 # Best model checkpoint
└── reports/
    └── metrics_cnn.json         # Training metrics & history
```

## 🔧 Development Workflow

### Using Claude Code

This project includes automation for development with Claude:

1. **Setup**: Extract upgrade package to project root
2. **Follow**: Instructions in `CLAUDE.md`
3. **Execute**: Prompts from `prompts/claude-commands.md`
4. **Specs**: Detailed specifications in `SPECS/`

### Model Development Guidelines

1. **Data Preparation** (`app/data/`):
   - Implement phase folding for light curves
   - Create dual views (global + local)
   - Robust normalization

2. **Model Architecture** (`app/models/`):
   - Follow two-branch pattern
   - Use batch normalization
   - Global average pooling for invariance

3. **Training** (`app/trainers/`):
   - Seed for reproducibility
   - Early stopping on validation AP
   - Save best model checkpoint
   - Log metrics to JSON

4. **Evaluation** (`app/calibration/`):
   - Calibration for probability estimates
   - Comprehensive metrics reporting

## 📊 Performance Benchmarks

### Simple Neural Network (MLP)
- Architecture: 256→64→1
- Training: ~3500 samples
- Early stopping: patience=30
- Metrics: Accuracy, F1, Precision, Recall

### Random Forest
- Best params: depth=8, estimators=200, split=9
- Cross-validation: 10-fold CV
- Feature importance analysis available

### Two-Branch 1D-CNN
- Primary metric: PR-AUC (Average Precision)
- Benefits:
  - Direct light curve analysis (no feature engineering)
  - Dual-scale pattern recognition
  - Better generalization on transit morphology
  - MPS acceleration on Apple Silicon

## 🛠️ Advanced Features

### Phase Folding (app/data/fold.py:10)
```python
phase_fold(t, period, t0)  # Fold time series by orbital period
```

### Robust Normalization (app/data/fold.py:21)
```python
robust_norm(x)  # MAD-based normalization (outlier-resistant)
```

### View Construction (app/data/fold.py:26)
```python
make_views(time, flux, period, t0, duration,
           g_len=2000, l_len=512, k=3.0)
# Returns: (global_view, local_view)
```

## 📝 Model Card

For production deployment, use the model card template in `SPECS/model_card_template.md` to document:
- Model details and intended use
- Training data and evaluation metrics
- Ethical considerations
- Limitations and biases

## 🔬 Research Background

This project is based on transit photometry analysis for exoplanet detection:

1. **Transit Method**: Detect periodic dips in stellar brightness when an orbiting planet passes in front of the star
2. **Kepler Mission**: NASA space telescope that monitored 150,000+ stars (2009-2018)
3. **KOI Classification**: Distinguish true planetary transits from false positives (eclipsing binaries, stellar variability, etc.)

## 🤝 Contributing

Development workflow:
1. Review specifications in `SPECS/`
2. Follow coding patterns in `app/`
3. Add tests for new features
4. Update documentation

## 📄 License

This project analyzes public NASA Kepler mission data.

## 🔗 References

- **Kepler Mission**: https://www.nasa.gov/kepler
- **KOI Catalog**: NASA Exoplanet Archive
- **TSFresh**: Time series feature extraction library
- **PyTorch**: Deep learning framework

## 📧 Contact

For questions about model implementation or data processing, refer to:
- `SPECS/` directory for detailed specifications
- `prompts/claude-commands.md` for development workflows
- Issue tracker (if applicable)

---

**Note**: This project demonstrates multiple ML approaches for exoplanet classification. The Two-Branch 1D-CNN represents the most modern approach with direct light curve analysis, while the feature-based models (RF, XGBoost) provide baseline comparisons.
