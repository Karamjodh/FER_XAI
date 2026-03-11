# 🧠 Explainable Facial Expression Recognition (XAI-FER)

A deep learning project for **7-class Facial Expression Recognition** with post-hoc explainability using **LIME** and **SHAP**. Two CNN architectures — ResNet-50 and EfficientNet-B0 — are trained and evaluated on FER2013 and RAF-DB, with pixel-level and region-level explanations generated for every prediction.

> Undergraduate Thesis Project · Department of Computer Science · 2026

---

## 📊 Results

### Overall Performance

| Model | Dataset | Accuracy | Macro F1 | Weighted F1 | Mean AUC |
|---|---|---|---|---|---|
| **ResNet-50** | **RAF-DB** | **84.36%** | **76.01%** | **84.45%** | **93.55%** |
| EfficientNet-B0 | RAF-DB | 82.25% | 71.99% | 82.61% | 91.35% |
| ResNet-50 | FER2013 | 70.88% | 69.45% | 70.69% | 91.60% |
| EfficientNet-B0 | FER2013 | 71.08% | 68.21% | 70.97% | 91.13% |
| AlexNet *(baseline)* | FER2013 | ~65.00% | ~61% | — | ~72% |

> ✅ Both models beat the baseline on FER2013. ResNet-50 on RAF-DB achieves **84.36%** — the best result overall.

---

### Per-Class F1 Scores — FER2013

| Emotion | ResNet-50 | EfficientNet-B0 |
|---|---|---|
| Angry | 64.37% | 64.02% |
| Disgust | 70.49% | 58.82% |
| Fear | 54.05% | 57.60% |
| **Happy** | **89.94%** | **89.55%** |
| Neutral | 68.99% | 69.88% |
| Sad | 55.64% | 55.77% |
| **Surprise** | **82.65%** | **81.82%** |

---

### Per-Class F1 Scores — RAF-DB

| Emotion | ResNet-50 | EfficientNet-B0 |
|---|---|---|
| Angry | 72.46% | 76.92% |
| Disgust | 59.02% | 52.94% |
| Fear | 54.55% | 37.50% |
| **Happy** | **91.65%** | **91.91%** |
| Neutral | 82.96% | 80.30% |
| Sad | 82.42% | 78.21% |
| **Surprise** | **89.04%** | **86.13%** |

---

## 📁 Project Structure

```
XAI-FER/
│
├── Config.py               # All hyperparameters, paths, and settings
├── Datasets.py             # FER2013 + RAF-DB dataset loaders
├── Models.py               # ResNet-50 and EfficientNet-B0 definitions
├── Train.py                # Single-phase training pipeline
├── Evaluate.py             # Accuracy, F1, AUC, confusion matrix
├── explain_lime.py         # LIME explanation generator
├── explain_shap.py         # SHAP explanation generator
├── report_generator.py     # HTML report generator
├── monitor.py              # GPU temperature monitor
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── fer2013/
│   │   └── fer2013.csv                  # 35,887 grayscale 48×48 images
│   └── rafdb/
│       └── DATASET/
│           ├── train/
│           │   ├── 1/  ← Surprise
│           │   ├── 2/  ← Fear
│           │   ├── 3/  ← Disgust
│           │   ├── 4/  ← Happy
│           │   ├── 5/  ← Sad
│           │   ├── 6/  ← Angry
│           │   └── 7/  ← Neutral
│           └── test/   ← same structure
│
└── outputs/
    ├── checkpoints/        # .pth weights + _history.json + _results.json
    ├── plots/              # Confusion matrix, ROC curves, training curves
    ├── explanations/
    │   ├── lime/           # LIME visualizations per model/dataset
    │   └── shap/           # SHAP visualizations per model/dataset
    └── reports/            # Generated HTML reports
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Karamjodh/FER_XAI.git
cd FER_XAI
```

### 2. Create Environment

```bash
conda create -n fer_xai python=3.9
conda activate fer_xai
pip install -r requirements.txt
```

### 3. Requirements

```
torch torchvision
numpy<2
pandas
Pillow
scikit-learn
matplotlib
lime
shap
scipy
tqdm
```

### 4. Hardware

Tested on:
- RTX 3050 Laptop GPU (4GB VRAM) — `NUM_WORKERS=0`, `BATCH_SIZE=64`
- RTX 3050 (6GB VRAM)

AMP (Automatic Mixed Precision) is enabled by default to reduce VRAM usage.

---

## 🗂️ Datasets

### FER2013
- **Source:** [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- 35,887 grayscale images · 48×48 pixels
- Pre-split: Training (28,709) · PublicTest (3,589) · PrivateTest (3,589)
- Heavily imbalanced — Happy: ~8,989 samples · Disgust: ~436 samples
- Place at: `data/fer2013/fer2013.csv`

### RAF-DB
- **Source:** [RAF-DB Basic](http://www.whdeng.cn/raf/model1.html)
- 29,672 real-world RGB face images · high resolution
- 12,271 train / 3,068 test (7-class basic emotion subset)
- Cleaner labels and higher image quality than FER2013
- Place at: `data/rafdb/DATASET/train/` and `data/rafdb/DATASET/test/`

### Unified Label Mapping

| Label Index | Emotion | FER2013 | RAF-DB Folder |
|---|---|---|---|
| 0 | Angry | 0 | 6 |
| 1 | Disgust | 1 | 3 |
| 2 | Fear | 2 | 2 |
| 3 | Happy | 3 | 4 |
| 4 | Neutral | 6 | 7 |
| 5 | Sad | 4 | 5 |
| 6 | Surprise | 5 | 1 |

---

## 🚀 Usage

### Training

```bash
# FER2013
python Train.py --model resnet50 --dataset fer2013
python Train.py --model efficientnet_b0 --dataset fer2013

# RAF-DB
python Train.py --model resnet50 --dataset rafdb
python Train.py --model efficientnet_b0 --dataset rafdb

# Train all models on a dataset
python Train.py --all --dataset fer2013
```

### Evaluation

```bash
python Evaluate.py --model resnet50 --dataset fer2013
python Evaluate.py --model efficientnet_b0 --dataset fer2013
python Evaluate.py --model resnet50 --dataset rafdb
python Evaluate.py --model efficientnet_b0 --dataset rafdb
```

### LIME Explanations

```bash
python explain_lime.py --model resnet50 --dataset fer2013 --samples 20 --lime-samples 1000
python explain_lime.py --model efficientnet_b0 --dataset fer2013 --samples 20 --lime-samples 1000
python explain_lime.py --model resnet50 --dataset rafdb --samples 20 --lime-samples 1000
python explain_lime.py --model efficientnet_b0 --dataset rafdb --samples 20 --lime-samples 1000
```

### SHAP Explanations

```bash
python explain_shap.py --model resnet50 --dataset fer2013 --samples 20 --bg-samples 100
python explain_shap.py --model efficientnet_b0 --dataset fer2013 --samples 20 --bg-samples 100
python explain_shap.py --model resnet50 --dataset rafdb --samples 20 --bg-samples 100
python explain_shap.py --model efficientnet_b0 --dataset rafdb --samples 20 --bg-samples 100
```

### Report Generation

```bash
python report_generator.py --dataset fer2013
python report_generator.py --dataset rafdb
```

---

## 🔍 Explainability

### LIME (Local Interpretable Model-Agnostic Explanations)
- Segments face into superpixels using QuickShift (`kernel_size=1.5`, `max_dist=20`, `ratio=0.2`)
- Generates 1,000 perturbed samples per image
- Highlights top-8 superpixel regions by importance weight
- **Output:** 4-panel figure — Original · Supporting Regions · All Regions · Importance Heatmap

### SHAP (SHapley Additive exPlanations)
- Uses `GradientExplainer` with 100 stratified background samples
- Computes Shapley values for all 7 classes simultaneously
- Enables class-contrastive analysis (which pixels push toward vs. away from each class)
- **Output:** 2-row figure — Row 1: Original · Pred SHAP · True SHAP · Absolute Importance · Row 2: All 7 class heatmaps

---

## 🏗️ Model Architecture

Both models use a **two-layer classification head** replacing the original FC layer:

```
Backbone (frozen ImageNet weights → fine-tuned)
    ↓
Linear(feature_dim → 512)
    ↓
ReLU + Dropout(0.4)
    ↓
Linear(512 → 7)
```

| Model | Backbone Feature Dim | Parameters |
|---|---|---|
| ResNet-50 | 2048 | ~25.6M |
| EfficientNet-B0 | 1280 | ~5.3M |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| LR Scheduler | CosineAnnealingLR (T_max=30) |
| Batch Size | 64 |
| Max Epochs | 30 |
| Early Stopping Patience | 8 |
| Label Smoothing | 0.1 |
| Loss | Weighted Cross-Entropy |
| AMP | Enabled |
| Training Strategy | Single-phase (no layer freezing) |

---

## 👥 Team

| Name | Role |
|---|---|
| Karamjodh | Model training · FER2013 · RAF-DB · LIME · SHAP · Evaluation · Report |

---

## 📄 License

This project is developed for academic research purposes as part of an undergraduate thesis.

---

## 🔗 References

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [RAF-DB Dataset](http://www.whdeng.cn/raf/model1.html)
- [LIME Paper](https://arxiv.org/abs/1602.04938) — Ribeiro et al., 2016
- [SHAP Paper](https://arxiv.org/abs/1705.07874) — Lundberg & Lee, 2017
- [ResNet Paper](https://arxiv.org/abs/1512.03385) — He et al., 2016
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019