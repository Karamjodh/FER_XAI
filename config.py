# ================================================================
#  config.py — Central Configuration
#  All hyperparameters, paths, and settings in one place.
#  Every other file imports from here.
# ================================================================

from pathlib import Path

# ----------------------------------------------------------------
# SECTION 1 — PATHS
# Path() is used instead of strings so it works on
# Windows, Linux, and Mac without changing anything.
# ----------------------------------------------------------------

BASE_DIR    = Path(__file__).parent        # Root of project folder
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
# FER2013 has 7 emotion classes.
# We define a UNIFIED order used across both datasets.
# ----------------------------------------------------------------

UNIFIED_CLASSES  = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES      = 7
IMG_SIZE         = 224    # All pretrained models expect 224x224 input

# ----------------------------------------------------------------
# SECTION 3 — MODEL SETTINGS
# These are the 3 models we will train and compare.
# ----------------------------------------------------------------

MODELS_TO_TRAIN = ["resnet50", "efficientnet_b0", "vgg16"]

# ----------------------------------------------------------------
# SECTION 4 — TRAINING HYPERPARAMETERS
# Tuned specifically for RTX 3050 (4GB VRAM)
# ----------------------------------------------------------------

BATCH_SIZE   = 32     # How many images per training step
NUM_EPOCHS   = 50     # Maximum epochs (early stopping will kick in earlier)
SEED         = 42     # For reproducibility — same results every run

# Two-phase fine-tuning:
# Phase 1 → freeze the backbone, only train the new head (fast, safe)
# Phase 2 → unfreeze last N layers, fine-tune with low LR (boosts accuracy)

PHASE1_EPOCHS   = 10
PHASE1_LR       = 1e-3    # Higher LR ok since only head is training

PHASE2_EPOCHS   = 40
PHASE2_LR       = 1e-4    # Must be low — we're touching pretrained weights
PHASE2_UNFREEZE = 30      # How many layers to unfreeze from the end

WEIGHT_DECAY         = 1e-4   # L2 regularization — prevents overfitting
EARLY_STOP_PATIENCE  = 8      # Stop if val_loss doesn't improve for 8 epochs

# ----------------------------------------------------------------
# SECTION 5 — AUGMENTATION SETTINGS
# Augmentation = artificially increasing dataset size by
# randomly transforming images during training only.
# ----------------------------------------------------------------

AUGMENT_TRAIN    = True
HORIZONTAL_FLIP  = True
RANDOM_ROTATION  = 15       # Rotate images up to ±15 degrees
COLOR_JITTER     = True     # Slight brightness/contrast changes
RANDOM_ERASING   = True     # Randomly erase small patches (helps with occlusion)

# ImageNet normalization — required for pretrained models
# These values are the mean and std of the ImageNet dataset
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------
# SECTION 6 — CLASS IMBALANCE
# FER2013 is heavily imbalanced — 'Happy' has 8000+ samples
# but 'Disgust' has only ~400. Weighted loss fixes this.
# ----------------------------------------------------------------

USE_WEIGHTED_LOSS = True

# ----------------------------------------------------------------
# SECTION 7 — EXPLAINABILITY SETTINGS
# ----------------------------------------------------------------

# LIME settings
LIME_NUM_SAMPLES  = 1000   # More = more stable but slower
LIME_NUM_FEATURES = 10     # Top N superpixel regions to highlight
LIME_NUM_SEGMENTS = 50     # How many superpixels to divide face into
LIME_BATCH_SIZE   = 64

# SHAP settings
SHAP_BACKGROUND_SAMPLES = 100
SHAP_TEST_SAMPLES       = 10

# ----------------------------------------------------------------
# SECTION 8 — HARDWARE SETTINGS
# ----------------------------------------------------------------

USE_AMP    = True   # Automatic Mixed Precision — uses float16
                    # Cuts VRAM usage by ~40% on RTX 3050. Always keep True.
NUM_WORKERS = 4     # Parallel data loading threads
PIN_MEMORY  = True  # Faster CPU→GPU data transfer

# ----------------------------------------------------------------
# SECTION 9 — LOGGING
# ----------------------------------------------------------------

LOG_INTERVAL = 10   # Print progress every 10 batches


# ----------------------------------------------------------------
# QUICK SANITY CHECK
# Run this file directly to confirm paths are correct:
# python config.py
# ----------------------------------------------------------------

OPTIMIZER  = "adam"      # options: "adam", "adamw"
SCHEDULER  = "cosine"    # options: "cosine", "step"
LR_STEP_SIZE = 10        # used if SCHEDULER = "step"
LR_GAMMA     = 0.5       # LR multiplier per step
LR_MIN       = 1e-6      # minimum LR for cosine scheduler

if __name__ == "__main__":
    print("BASE_DIR :", BASE_DIR)
    print("DATA_DIR :", DATA_DIR)
    print("CKPT_DIR :", CKPT_DIR)
    print("Classes  :", UNIFIED_CLASSES)
    print("Models   :", MODELS_TO_TRAIN)
    print("\n✅ config.py loaded successfully")