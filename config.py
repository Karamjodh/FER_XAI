# ================================================================
#  Config.py — Central Configuration
#  All hyperparameters, paths, and settings in one place.
#  Every other file imports from here.
# ================================================================

from pathlib import Path

# ----------------------------------------------------------------
# SECTION 1 — PATHS
# ----------------------------------------------------------------

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
FER_DIR     = DATA_DIR / "fer2013"
RAFDB_DIR   = DATA_DIR / "rafdb"

OUTPUTS_DIR = BASE_DIR / "outputs"
CKPT_DIR    = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
EXPLAIN_DIR = OUTPUTS_DIR / "explanations"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# ----------------------------------------------------------------
# SECTION 2 — DATASET SETTINGS
# ----------------------------------------------------------------

UNIFIED_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES     = 7
IMG_SIZE        = 224

# ----------------------------------------------------------------
# SECTION 3 — MODEL SETTINGS
# ----------------------------------------------------------------

MODELS_TO_TRAIN = ["resnet50", "efficientnet_b0"]

# ----------------------------------------------------------------
# SECTION 4 — TRAINING HYPERPARAMETERS
#
# Single-phase training (no freezing).
# Friend's approach: train everything from epoch 1 with low LR.
# Got 75% vs our 69% with 2-phase. Simpler and better.
# ----------------------------------------------------------------

BATCH_SIZE          = 64      # increased from 16 → more stable gradients
SEED                = 42
EPOCHS              = 30      # friend used 30, clean and sufficient
LR                  = 1e-4   # low enough to safely train pretrained weights
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 8       # stop if val_loss stagnates

# Label smoothing — well known to improve 1-3% on noisy datasets like FER2013
LABEL_SMOOTHING = 0.1

# ----------------------------------------------------------------
# SECTION 5 — AUGMENTATION SETTINGS
# Removed RandomErasing — it was erasing eyes/mouth on face images
# ----------------------------------------------------------------

AUGMENT_TRAIN   = True
HORIZONTAL_FLIP = True
RANDOM_ROTATION = 10          # friend used 10, we had 15
COLOR_JITTER    = True
RANDOM_ERASING  = False       # REMOVED — was hurting facial feature learning

# ImageNet normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------
# SECTION 6 — CLASS IMBALANCE
# ----------------------------------------------------------------

USE_WEIGHTED_LOSS = True

# ----------------------------------------------------------------
# SECTION 7 — EXPLAINABILITY SETTINGS
# ----------------------------------------------------------------

LIME_NUM_SAMPLES  = 1000
LIME_NUM_FEATURES = 10
LIME_NUM_SEGMENTS = 50
LIME_BATCH_SIZE   = 64

SHAP_BACKGROUND_SAMPLES = 100
SHAP_TEST_SAMPLES       = 10

# ----------------------------------------------------------------
# SECTION 8 — HARDWARE SETTINGS
# ----------------------------------------------------------------

USE_AMP     = True
NUM_WORKERS = 0
PIN_MEMORY  = True

# ----------------------------------------------------------------
# SECTION 9 — OPTIMIZER & SCHEDULER
# CosineAnnealingLR smoothly decays LR — better than CyclicLR
# which oscillates and hurts final convergence
# ----------------------------------------------------------------

OPTIMIZER  = "adamw"
LR_MIN     = 1e-6     # minimum LR at end of cosine schedule

# ----------------------------------------------------------------
# SECTION 10 — LOGGING
# ----------------------------------------------------------------

LOG_INTERVAL = 10


if __name__ == "__main__":
    print("BASE_DIR :", BASE_DIR)
    print("DATA_DIR :", DATA_DIR)
    print("CKPT_DIR :", CKPT_DIR)
    print("Classes  :", UNIFIED_CLASSES)
    print("Models   :", MODELS_TO_TRAIN)
    print("\n✅ Config.py loaded successfully")