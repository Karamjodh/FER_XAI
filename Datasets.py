import numpy as np
import pandas as pd
from PIL import Image 
from pathlib import Path 
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T

from Config import (
    FER_DIR, RAFDB_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    NORMALIZE_MEAN, NORMALIZE_STD, AUGMENT_TRAIN, HORIZONTAL_FLIP,
    RANDOM_ROTATION, COLOR_JITTER, RANDOM_ERASING, UNIFIED_CLASSES, SEED
)

def get_transforms(split: str) -> T.Compose:
    if split == 'train' and AUGMENT_TRAIN:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.Grayscale(num_output_channels=3),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=RANDOM_ROTATION),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            T.RandomErasing(p=0.2)
        ])
    
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


# ─────────────────────────────────────────────
#  FER2013
# ─────────────────────────────────────────────
class FER2013Dataset(Dataset):
    # Map FER2013 original labels → unified 7-class labels
    FER_TO_UNIFIED = {
        0: 0,  # Angry
        1: 1,  # Disgust
        2: 2,  # Fear
        3: 3,  # Happy
        4: 5,  # Sad
        5: 6,  # Surprise
        6: 4,  # Neutral
    }

    def __init__(self, split: str = "train", transform=None):
        split_map = {
            "train": "Training",
            "val":   "PublicTest",
            "test":  "PrivateTest"
        }
        csv_path = FER_DIR / "fer2013.csv"
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split_map[split]].reset_index(drop=True)
        self.pixels = df["pixels"].values
        self.labels = df["emotion"].values
        self.transform = transform
        self.split = split
        print(f"FER2013[{split}] : {len(self.labels)} samples")
        print(f"Class distribution : {dict(Counter(self.labels))}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pixel_vals = np.array(self.pixels[idx].split(), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixel_vals, mode="L")
        label = self.FER_TO_UNIFIED[int(self.labels[idx])]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        unified_labels = [self.FER_TO_UNIFIED[l] for l in self.labels]
        counts = Counter(unified_labels)
        total = sum(counts.values())
        weights = torch.zeros(7)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (7 * count)
        return weights


# ─────────────────────────────────────────────
#  RAF-DB
# ─────────────────────────────────────────────
class RAFDBDataset(Dataset):
    """
    RAF-DB dataset loader.
    
    Expected structure:
        C:/rafdb/
        ├── DATASET/
        │   ├── train/   ← image files (e.g. train_00001_aligned.jpg)
        │   └── test/    ← image files (e.g. test_0001_aligned.jpg)
        ├── train_labels.csv   ← columns: image, label (1-7)
        └── test_labels.csv    ← columns: image, label (1-7)

    RAF-DB label mapping (1-indexed) → Unified labels (0-indexed):
        1 → Surprise  (6)
        2 → Fear      (2)
        3 → Disgust   (1)
        4 → Happy     (3)
        5 → Sad       (5)
        6 → Angry     (0)
        7 → Neutral   (4)
    """

    # RAF-DB labels are 1-indexed → map to unified 0-indexed
    RAFDB_TO_UNIFIED = {
        1: 6,  # Surprise
        2: 2,  # Fear
        3: 1,  # Disgust
        4: 3,  # Happy
        5: 5,  # Sad
        6: 0,  # Angry
        7: 4,  # Neutral
    }

    def __init__(self, split: str = "train", transform=None):
        self.transform = transform
        self.split = split

        # RAF-DB only has train/test (no separate val in CSV)
        # We use test CSV for both val and test splits
        if split == "train":
            csv_path   = RAFDB_DIR / "train_labels.csv"
            img_folder = RAFDB_DIR / "DATASET" / "train"
        else:
            # val and test both use the test set
            csv_path   = RAFDB_DIR / "test_labels.csv"
            img_folder = RAFDB_DIR / "DATASET" / "test"

        df = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.image_names = df["image"].values
        self.labels      = df["label"].values

        # If val split → use first 80%, test → last 20%
        if split == "val":
            cut = int(0.8 * len(self.image_names))
            self.image_names = self.image_names[:cut]
            self.labels      = self.labels[:cut]
        elif split == "test":
            cut = int(0.8 * len(self.image_names))
            self.image_names = self.image_names[cut:]
            self.labels      = self.labels[cut:]

        print(f"RAFDB[{split}] : {len(self.labels)} samples")
        unified = [self.RAFDB_TO_UNIFIED[l] for l in self.labels]
        print(f"Class distribution : {dict(Counter(unified))}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_folder / self.image_names[idx]
        image    = Image.open(img_path).convert("RGB")
        label    = self.RAFDB_TO_UNIFIED[int(self.labels[idx])]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        unified_labels = [self.RAFDB_TO_UNIFIED[l] for l in self.labels]
        counts = Counter(unified_labels)
        total  = sum(counts.values())
        weights = torch.zeros(7)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (7 * count)
        return weights


# ─────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────
def get_dataloaders(dataset_name: str = "fer2013"):
    print(f"\nLoading {dataset_name.upper()}")

    if dataset_name == "fer2013":
        train_ds = FER2013Dataset("train", transform=get_transforms("train"))
        val_ds   = FER2013Dataset("val",   transform=get_transforms("val"))
        test_ds  = FER2013Dataset("test",  transform=get_transforms("test"))

    elif dataset_name == "rafdb":
        train_ds = RAFDBDataset("train", transform=get_transforms("train"))
        val_ds   = RAFDBDataset("val",   transform=get_transforms("val"))
        test_ds  = RAFDBDataset("test",  transform=get_transforms("test"))

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Choose 'fer2013' or 'rafdb'.")

    class_weights = train_ds.get_class_weights()

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"Train Batches : {len(train_loader)}")
    print(f"Val Batches   : {len(val_loader)}")
    print(f"Test Batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights