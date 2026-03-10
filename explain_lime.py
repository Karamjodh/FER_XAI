import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

from Config import (
    CKPT_DIR, UNIFIED_CLASSES, NUM_CLASSES, IMG_SIZE, SEED
)
from Models import get_model, load_checkpoint

# ── Settings ────────────────────────────────────────────────────
NUM_SAMPLES      = 1000
NUM_FEATURES     = 8
BATCH_SIZE_LIME  = 32
EXPLANATIONS_DIR = Path("outputs/explanations/lime")
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Transform ────────────────────────────────────────────────────
def get_lime_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ── Predict function ─────────────────────────────────────────────
def make_predict_fn(model, device):
    transform = get_lime_transform()

    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = torch.stack([
            transform(Image.fromarray(img.astype(np.uint8)))
            for img in images
        ]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn

# ── Load FER2013 test samples ─────────────────────────────────────
def load_fer2013_samples(num_samples: int = 20):
    import pandas as pd
    from Config import FER_DIR

    csv_path = FER_DIR / "fer2013.csv"
    df       = pd.read_csv(csv_path)
    test_df  = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    FER_TO_UNIFIED = {0:0, 1:1, 2:2, 3:3, 4:5, 5:6, 6:4}

    if num_samples < NUM_CLASSES:
        selected_df = test_df.sample(num_samples, random_state=SEED)
    else:
        samples_per_class = num_samples // NUM_CLASSES
        selected = []
        for class_idx in range(NUM_CLASSES):
            class_rows = test_df[test_df["emotion"] == class_idx]
            n = min(samples_per_class, len(class_rows))
            selected.append(class_rows.sample(n, random_state=SEED))
        selected_df = pd.concat(selected).reset_index(drop=True)

    images, labels = [], []
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

    for _, row in selected_df.iterrows():
        pixels = np.array(
            row["pixels"].split(), dtype=np.uint8
        ).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L").convert("RGB")
        img = resize(img)
        images.append(img)
        labels.append(FER_TO_UNIFIED[int(row["emotion"])])

    print(f"  Loaded {len(images)} FER2013 test samples")
    return images, labels

# ── Load RAF-DB test samples ──────────────────────────────────────
def load_rafdb_samples(num_samples: int = 20):
    from Config import RAFDB_DIR

    RAFDB_TO_UNIFIED = {
        1: 6,  # Surprise
        2: 2,  # Fear
        3: 1,  # Disgust
        4: 3,  # Happy
        5: 5,  # Sad
        6: 0,  # Angry
        7: 4,  # Neutral
    }

    test_folder = RAFDB_DIR / "DATASET" / "test"
    samples_per_class = max(1, num_samples // NUM_CLASSES)

    images, labels = [], []
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    random.seed(SEED)

    for folder_id in range(1, 8):
        folder = test_folder / str(folder_id)
        if not folder.exists():
            print(f"  ⚠ Folder not found: {folder}")
            continue

        imgs = list(folder.glob("*.jpg"))  + \
               list(folder.glob("*.png"))  + \
               list(folder.glob("*.jpeg"))
        random.shuffle(imgs)
        chosen = imgs[:samples_per_class]

        for img_path in chosen:
            img = Image.open(img_path).convert("RGB")
            img = resize(img)
            images.append(img)
            labels.append(RAFDB_TO_UNIFIED[folder_id])

    print(f"  Loaded {len(images)} RAF-DB test samples")
    return images, labels

# ── Unified loader ────────────────────────────────────────────────
def load_test_samples(dataset_name: str, num_samples: int = 20):
    if dataset_name == "fer2013":
        return load_fer2013_samples(num_samples)
    elif dataset_name == "rafdb":
        return load_rafdb_samples(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# ── LIME explanation ──────────────────────────────────────────────
def generate_lime_explanation(image: Image.Image, predict_fn, num_samples: int = NUM_SAMPLES):
    explainer = lime_image.LimeImageExplainer()
    img_array = np.array(image)

    segmenter = SegmentationAlgorithm(
        "quickshift",
        kernel_size = 1.5,
        max_dist    = 20,
        ratio       = 0.2
    )

    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels      = NUM_CLASSES,
        hide_color      = 0,
        num_samples     = num_samples,
        segmentation_fn = segmenter,
        batch_size      = BATCH_SIZE_LIME
    )
    return explanation, img_array

# ── Visualize ────────────────────────────────────────────────────
def visualize_explanation(
    explanation,
    img_array:    np.ndarray,
    true_label:   int,
    pred_label:   int,
    pred_probs:   np.ndarray,
    save_path:    Optional[Path] = None,
    num_features: int = NUM_FEATURES,
):
    pred_label = int(pred_label)
    true_label = int(true_label)

    available_labels = list(explanation.local_exp.keys())
    if pred_label not in available_labels:
        pred_label = available_labels[0]

    correct = pred_label == true_label

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0f0f0f')
    for ax in axes:
        ax.set_facecolor('#0f0f0f')

    # ── Panel 1: Original ──
    axes[0].imshow(img_array)
    axes[0].set_title(
        f"Original\nTrue: {UNIFIED_CLASSES[true_label]}",
        fontsize=11, fontweight="bold", color='white'
    )
    axes[0].axis("off")

    # ── Panel 2: Positive regions only ──
    temp, mask = explanation.get_image_and_mask(
        pred_label,
        positive_only = True,
        num_features  = num_features,
        hide_rest     = True
    )
    axes[1].imshow(mark_boundaries(
        temp / 255.0 if temp.max() > 1 else temp,
        mask, color=(1, 0.8, 0)
    ))
    axes[1].set_title(
        f"Supporting Regions\nPred: {UNIFIED_CLASSES[pred_label]} ({pred_probs[pred_label]:.1%})",
        fontsize=11, fontweight="bold",
        color='#00ff88' if correct else '#ff4444'
    )
    axes[1].axis("off")

    # ── Panel 3: All contributing regions ──
    temp2, mask2 = explanation.get_image_and_mask(
        pred_label,
        positive_only = False,
        num_features  = num_features,
        hide_rest     = True
    )
    axes[2].imshow(mark_boundaries(
        temp2 / 255.0 if temp2.max() > 1 else temp2,
        mask2
    ))
    axes[2].set_title(
        "All Regions\nGreen=For | Red=Against",
        fontsize=11, color='white'
    )
    axes[2].axis("off")

    # ── Panel 4: Importance heatmap ──
    segments     = explanation.segments
    lime_weights = dict(explanation.local_exp[pred_label])
    heatmap      = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in lime_weights.items():
        heatmap[segments == seg_id] = weight

    if heatmap.max() != heatmap.min():
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap_norm = heatmap

    axes[3].imshow(img_array, alpha=0.5)
    im = axes[3].imshow(heatmap_norm, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title(
        "Importance Heatmap\nRed=High | Green=Low",
        fontsize=11, color='white'
    )
    axes[3].axis("off")

    # ── Probability bar ──
    prob_text = " | ".join([
        f"{UNIFIED_CLASSES[i]}: {pred_probs[i]:.1%}"
        for i in range(NUM_CLASSES)
    ])
    fig.suptitle(
        f"LIME: True={UNIFIED_CLASSES[true_label]} | "
        f"Pred={UNIFIED_CLASSES[pred_label]} | "
        f"{'✓ Correct' if correct else '✗ Wrong'}\n{prob_text}",
        fontsize=9, color='white', y=0.02
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor='#0f0f0f')
        plt.close()
    else:
        plt.show()

    return fig

# ── Main explain function ─────────────────────────────────────────
def explain_model(
    model_name:   str,
    dataset_name: str = "fer2013",
    num_samples:  int = 20,
    lime_samples: int = NUM_SAMPLES,
):
    print(f"\n{'='*60}")
    print(f"LIME EXPLANATION : {model_name} on {dataset_name}")
    print(f"{'='*60}")

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return None

    model = get_model(model_name, pretrained=False).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()

    predict_fn = make_predict_fn(model, device)

    print("  Loading test samples...")
    images, labels = load_test_samples(dataset_name, num_samples)

    out_dir = EXPLANATIONS_DIR / f"{model_name}_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for idx, (image, true_label) in enumerate(zip(images, labels)):
        print(f"\n  [{idx+1}/{len(images)}] True: {UNIFIED_CLASSES[true_label]}")

        transform = get_lime_transform()
        tensor    = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_label = int(np.argmax(probs))
        correct    = pred_label == true_label

        print(f"  Pred: {UNIFIED_CLASSES[pred_label]} ({probs[pred_label]:.1%}) "
              f"{'✓' if correct else '✗'}")
        print(f"  Generating LIME ({lime_samples} perturbations)...")

        explanation, img_array = generate_lime_explanation(
            image, predict_fn, lime_samples
        )

        save_path = out_dir / \
            f"{idx:03d}_{UNIFIED_CLASSES[true_label]}_" \
            f"pred{UNIFIED_CLASSES[pred_label]}.png"

        visualize_explanation(
            explanation, img_array,
            int(true_label), int(pred_label),
            probs, save_path=save_path,
            num_features=NUM_FEATURES
        )
        print(f"  Saved → {save_path.name}")

        summary.append({
            "idx":        idx,
            "true":       UNIFIED_CLASSES[true_label],
            "pred":       UNIFIED_CLASSES[pred_label],
            "confidence": float(probs[pred_label]),
            "correct":    correct,
            "saved":      str(save_path)
        })

    summary_path = out_dir / "lime_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    correct_count = sum(s["correct"] for s in summary)
    print(f"\n  Done! {correct_count}/{len(summary)} correct")
    print(f"  Summary → {summary_path}")
    print(f"  Images  → {out_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--dataset",      type=str, default="fer2013",
                        choices=["fer2013", "rafdb"])
    parser.add_argument("--samples",      type=int, default=20)
    parser.add_argument("--lime-samples", type=int, default=1000)
    args = parser.parse_args()

    explain_model(args.model, args.dataset, args.samples, args.lime_samples)