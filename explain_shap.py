import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import shap
from scipy.ndimage import gaussian_filter

from Config import (
    CKPT_DIR, UNIFIED_CLASSES, NUM_CLASSES, IMG_SIZE, SEED
)
from Models import get_model, load_checkpoint

# ── Settings ─────────────────────────────────────────────────────
BACKGROUND_SAMPLES = 100
SHAP_TEST_SAMPLES  = 20
EXPLANATIONS_DIR   = Path("outputs/explanations/shap")
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ── FER2013 loaders ───────────────────────────────────────────────
def load_fer2013_background(num_samples: int):
    import pandas as pd
    from Config import FER_DIR

    df       = pd.read_csv(FER_DIR / "fer2013.csv")
    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)

    FER_TO_UNIFIED = {0:0, 1:1, 2:2, 3:3, 4:5, 5:6, 6:4}
    samples_per_class = max(1, num_samples // NUM_CLASSES)
    selected = []
    for class_idx in range(NUM_CLASSES):
        # class_idx here is FER original label
        class_rows = train_df[train_df["emotion"] == class_idx]
        n = min(samples_per_class, len(class_rows))
        selected.append(class_rows.sample(n, random_state=SEED))
    selected_df = pd.concat(selected).reset_index(drop=True)

    tensors = []
    for _, row in selected_df.iterrows():
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img    = Image.fromarray(pixels, mode="L").convert("RGB")
        tensors.append(TRANSFORM(img))

    background = torch.stack(tensors)
    print(f"  FER2013 background: {background.shape}")
    return background


def load_fer2013_test(num_samples: int):
    import pandas as pd
    from Config import FER_DIR

    FER_TO_UNIFIED = {0:0, 1:1, 2:2, 3:3, 4:5, 5:6, 6:4}

    df      = pd.read_csv(FER_DIR / "fer2013.csv")
    test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    samples_per_class = max(1, num_samples // NUM_CLASSES)
    selected = []
    for class_idx in range(NUM_CLASSES):
        class_rows = test_df[test_df["emotion"] == class_idx]
        n = min(samples_per_class, len(class_rows))
        selected.append(class_rows.sample(n, random_state=SEED))
    selected_df = pd.concat(selected).reset_index(drop=True)

    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    images, tensors, labels = [], [], []

    for _, row in selected_df.iterrows():
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img    = Image.fromarray(pixels, mode="L").convert("RGB")
        img    = resize(img)
        images.append(img)
        tensors.append(TRANSFORM(img))
        labels.append(FER_TO_UNIFIED[int(row["emotion"])])

    print(f"  FER2013 test samples: {len(images)}")
    return images, torch.stack(tensors), labels


# ── RAF-DB loaders ────────────────────────────────────────────────
RAFDB_TO_UNIFIED = {1:6, 2:2, 3:1, 4:3, 5:5, 6:0, 7:4}

def load_rafdb_background(num_samples: int):
    from Config import RAFDB_DIR

    train_folder      = RAFDB_DIR / "DATASET" / "train"
    samples_per_class = max(1, num_samples // NUM_CLASSES)
    tensors           = []

    random.seed(SEED)
    for folder_id in range(1, 8):
        folder = train_folder / str(folder_id)
        if not folder.exists():
            continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        random.shuffle(imgs)
        for img_path in imgs[:samples_per_class]:
            img = Image.open(img_path).convert("RGB")
            tensors.append(TRANSFORM(img))

    background = torch.stack(tensors)
    print(f"  RAF-DB background: {background.shape}")
    return background


def load_rafdb_test(num_samples: int):
    from Config import RAFDB_DIR

    test_folder       = RAFDB_DIR / "DATASET" / "test"
    samples_per_class = max(1, num_samples // NUM_CLASSES)
    resize            = transforms.Resize((IMG_SIZE, IMG_SIZE))
    images, tensors, labels = [], [], []

    random.seed(SEED)
    for folder_id in range(1, 8):
        folder = test_folder / str(folder_id)
        if not folder.exists():
            continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        random.shuffle(imgs)
        for img_path in imgs[:samples_per_class]:
            img = Image.open(img_path).convert("RGB")
            img = resize(img)
            images.append(img)
            tensors.append(TRANSFORM(img))
            labels.append(RAFDB_TO_UNIFIED[folder_id])

    print(f"  RAF-DB test samples: {len(images)}")
    return images, torch.stack(tensors), labels


# ── Unified routers ───────────────────────────────────────────────
def load_background_samples(dataset_name: str, num_samples: int):
    if dataset_name == "fer2013":
        return load_fer2013_background(num_samples)
    elif dataset_name == "rafdb":
        return load_rafdb_background(num_samples)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def load_test_samples(dataset_name: str, num_samples: int):
    if dataset_name == "fer2013":
        return load_fer2013_test(num_samples)
    elif dataset_name == "rafdb":
        return load_rafdb_test(num_samples)
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Disable inplace ReLU (required for SHAP) ─────────────────────
def disable_inplace(model):
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return model


# ── Compute SHAP values ───────────────────────────────────────────
def generate_shap_values(model, background, test_tensors, device):
    print("\n  Computing SHAP values...")
    background   = background.to(device)
    test_tensors = test_tensors.to(device)

    explainer        = shap.GradientExplainer(model, background)
    shap_values_list = []

    for i in range(len(test_tensors)):
        print(f"  SHAP [{i+1}/{len(test_tensors)}]...", end="\r")
        sv = explainer.shap_values(test_tensors[i:i+1])
        shap_values_list.append(sv)

    print(f"\n  All SHAP values computed!")
    stacked     = np.concatenate(shap_values_list, axis=0)
    shap_values = [stacked[..., c] for c in range(NUM_CLASSES)]
    print(f"  Shape per class: {shap_values[0].shape}")
    return shap_values


# ── Helpers ───────────────────────────────────────────────────────
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * std + mean, 0, 1)


def get_heatmap(shap_values, class_idx, sample_idx):
    sv = shap_values[class_idx][sample_idx].mean(axis=0)
    return gaussian_filter(sv, sigma=3)


def show_heatmap(ax, img_np, sv, title, title_color="white",
                 show_colorbar=False, fig=None):
    vmax = max(np.percentile(np.abs(sv), 95), 1e-8)
    ax.imshow(img_np)
    im = ax.imshow(sv, cmap="RdBu_r", alpha=0.65, vmin=-vmax, vmax=vmax)
    if show_colorbar and fig is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=8, color=title_color, fontweight="bold")
    ax.axis("off")
    return im


# ── Visualize ─────────────────────────────────────────────────────
def visualize_shap(shap_values, img_tensor, img_pil, true_label,
                   pred_label, pred_probs, sample_idx, save_path=None):
    true_label = int(true_label)
    pred_label = int(pred_label)
    correct    = pred_label == true_label
    img_np     = denormalize(img_tensor)

    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3,
                            top=0.88, bottom=0.12)

    # Row 1 Col 1: Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title(f"Original\nTrue: {UNIFIED_CLASSES[true_label]}",
                 fontsize=11, fontweight="bold", color="white")
    ax.axis("off")

    # Row 1 Col 2: Predicted class SHAP
    ax = fig.add_subplot(gs[0, 1])
    sv = get_heatmap(shap_values, pred_label, sample_idx)
    show_heatmap(ax, img_np, sv,
                 title=f"SHAP: {UNIFIED_CLASSES[pred_label]}\nPred ({pred_probs[pred_label]:.1%})",
                 title_color="#00ff88" if correct else "#ff4444",
                 show_colorbar=True, fig=fig)

    # Row 1 Col 3: True class SHAP
    ax = fig.add_subplot(gs[0, 2])
    sv = get_heatmap(shap_values, true_label, sample_idx)
    show_heatmap(ax, img_np, sv,
                 title=f"SHAP: {UNIFIED_CLASSES[true_label]}\nTrue ({pred_probs[true_label]:.1%})",
                 title_color="#88aaff")

    # Row 1 Col 4: Absolute importance
    ax  = fig.add_subplot(gs[0, 3])
    sv  = get_heatmap(shap_values, pred_label, sample_idx)
    abs_sv = np.abs(sv)
    abs_sv = (abs_sv - abs_sv.min()) / (abs_sv.max() - abs_sv.min() + 1e-8)
    ax.imshow(img_np, alpha=0.4)
    ax.imshow(abs_sv, cmap="hot", alpha=0.7)
    ax.set_title("Pixel Importance\n(Absolute SHAP)", fontsize=9,
                 color="white", fontweight="bold")
    ax.axis("off")

    # Row 2: All 7 classes
    gs2 = gridspec.GridSpecFromSubplotSpec(1, NUM_CLASSES,
                                           subplot_spec=gs[1, :], wspace=0.15)
    for class_idx in range(NUM_CLASSES):
        ax   = fig.add_subplot(gs2[class_idx])
        sv   = get_heatmap(shap_values, class_idx, sample_idx)
        is_pred = class_idx == pred_label
        is_true = class_idx == true_label
        if is_true and is_pred:  t_color = "#00ff88"
        elif is_true:            t_color = "#88aaff"
        elif is_pred:            t_color = "#ff4444"
        else:                    t_color = "#aaaaaa"

        show_heatmap(ax, img_np, sv,
                     title=f"{UNIFIED_CLASSES[class_idx]}\n{pred_probs[class_idx]:.1%}",
                     title_color=t_color)
        if is_pred or is_true:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(t_color)
                spine.set_linewidth(2)

    status = "✓ Correct" if correct else "✗ Wrong"
    fig.suptitle(
        f"SHAP  |  True: {UNIFIED_CLASSES[true_label]}  |  "
        f"Pred: {UNIFIED_CLASSES[pred_label]}  |  {status}",
        fontsize=13, fontweight="bold", color="white", y=0.97
    )
    fig.text(0.5, 0.03,
             "🔴 Red = pushes toward predicted class   "
             "🔵 Blue = pushes away   "
             "🟠 Hot = high absolute importance",
             ha="center", fontsize=8, color="#aaaaaa", style="italic")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.close()
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────
def explain_model(model_name, dataset_name="fer2013",
                  num_samples=SHAP_TEST_SAMPLES, bg_samples=BACKGROUND_SAMPLES):
    print(f"\n{'='*60}")
    print(f"  SHAP: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return None

    model = get_model(model_name, pretrained=False).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()
    model = disable_inplace(model)

    print("\n  Loading background samples...")
    background = load_background_samples(dataset_name, bg_samples)

    print("\n  Loading test samples...")
    images, tensors, labels = load_test_samples(dataset_name, num_samples)

    print("\n  Getting predictions...")
    with torch.no_grad():
        logits = model(tensors.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
    pred_labels = np.argmax(probs, axis=1)

    shap_values = generate_shap_values(model, background, tensors, device)

    out_dir = EXPLANATIONS_DIR / f"{model_name}_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    print("\n  Generating visualizations...")
    for idx in range(len(images)):
        true_label = int(labels[idx])
        pred_label = int(pred_labels[idx])
        correct    = pred_label == true_label

        print(f"  [{idx+1}/{len(images)}] "
              f"True: {UNIFIED_CLASSES[true_label]} | "
              f"Pred: {UNIFIED_CLASSES[pred_label]} "
              f"({'✓' if correct else '✗'})")

        save_path = out_dir / \
            f"{idx:03d}_{UNIFIED_CLASSES[true_label]}_pred{UNIFIED_CLASSES[pred_label]}.png"

        visualize_shap(shap_values, tensors[idx], images[idx],
                       true_label, pred_label, probs[idx],
                       sample_idx=idx, save_path=save_path)

        summary.append({
            "idx":        idx,
            "true":       UNIFIED_CLASSES[true_label],
            "pred":       UNIFIED_CLASSES[pred_label],
            "confidence": float(probs[idx][pred_label]),
            "correct":    correct,
            "saved":      str(save_path)
        })

    summary_path = out_dir / "shap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    correct_count = sum(s["correct"] for s in summary)
    print(f"\n  Done! {correct_count}/{len(summary)} correct")
    print(f"  Summary → {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--dataset",    type=str, default="fer2013",
                        choices=["fer2013", "rafdb"])
    parser.add_argument("--samples",    type=int, default=20)
    parser.add_argument("--bg-samples", type=int, default=100)
    parser.add_argument("--all",        action="store_true")
    args = parser.parse_args()

    if args.all:
        from Config import MODELS_TO_TRAIN
        for m in MODELS_TO_TRAIN:
            explain_model(m, args.dataset, args.samples, args.bg_samples)
    else:
        explain_model(args.model, args.dataset, args.samples, args.bg_samples)