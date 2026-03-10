import json
import argparse
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
    CKPT_DIR, UNIFIED_CLASSES, NUM_CLASSES, IMG_SIZE
)
from Models import get_model, load_checkpoint

# ── Settings ─────────────────────────────────────────────────────
BACKGROUND_SAMPLES = 100
SHAP_TEST_SAMPLES  = 20
EXPLANATIONS_DIR   = Path("outputs/explanations/shap")
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load background samples ───────────────────────────────────────
def load_background_samples(dataset_name: str, num_samples: int = BACKGROUND_SAMPLES):
    import pandas as pd
    from Config import FER_DIR

    csv_path = FER_DIR / "fer2013.csv"
    df       = pd.read_csv(csv_path)
    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)

    # Sample evenly across classes for better background diversity
    samples_per_class = max(1, num_samples // NUM_CLASSES)
    selected = []
    for class_idx in range(NUM_CLASSES):
        class_rows = train_df[train_df["emotion"] == class_idx]
        n = min(samples_per_class, len(class_rows))
        selected.append(class_rows.sample(n, random_state=42))
    selected_df = pd.concat(selected).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensors = []
    for _, row in selected_df.iterrows():
        pixels = np.array(
            row["pixels"].split(), dtype=np.uint8
        ).reshape(48, 48)
        img    = Image.fromarray(pixels, mode="L").convert("RGB")
        tensors.append(transform(img))

    background = torch.stack(tensors)
    print(f"  Background samples: {background.shape} "
          f"({samples_per_class} per class)")
    return background

# ── Load test samples ─────────────────────────────────────────────
def load_test_samples(dataset_name: str, num_samples: int = SHAP_TEST_SAMPLES):
    import pandas as pd
    from Config import FER_DIR

    csv_path = FER_DIR / "fer2013.csv"
    df       = pd.read_csv(csv_path)
    test_df  = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    samples_per_class = max(1, num_samples // NUM_CLASSES)
    selected = []
    for class_idx in range(NUM_CLASSES):
        class_rows = test_df[test_df["emotion"] == class_idx]
        n = min(samples_per_class, len(class_rows))
        selected.append(class_rows.sample(n, random_state=42))
    selected_df = pd.concat(selected).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    resize  = transforms.Resize((IMG_SIZE, IMG_SIZE))
    images, tensors, labels = [], [], []

    for _, row in selected_df.iterrows():
        pixels = np.array(
            row["pixels"].split(), dtype=np.uint8
        ).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L").convert("RGB")
        img = resize(img)
        images.append(img)
        tensors.append(transform(img))
        labels.append(int(row["emotion"]))

    tensors = torch.stack(tensors)
    print(f"  Test samples: {tensors.shape}")
    return images, tensors, labels

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

    explainer = shap.GradientExplainer(model, background)

    shap_values_list = []
    for i in range(len(test_tensors)):
        print(f"  SHAP [{i+1}/{len(test_tensors)}]...", end="\r")
        sv = explainer.shap_values(test_tensors[i:i+1])
        shap_values_list.append(sv)

    print(f"\n  All SHAP values computed!")

    # Stack → (N, 3, 224, 224, 7)
    stacked     = np.concatenate(shap_values_list, axis=0)
    # Split into list of 7 arrays each (N, 3, 224, 224)
    shap_values = [stacked[..., c] for c in range(NUM_CLASSES)]

    print(f"  Shape per class: {shap_values[0].shape}")
    return shap_values

# ── Denormalize image for display ────────────────────────────────
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = np.clip(img * std + mean, 0, 1)
    return img

# ── Get clean heatmap from SHAP values ───────────────────────────
def get_heatmap(shap_values, class_idx, sample_idx):
    sv = shap_values[class_idx][sample_idx]  # (3, 224, 224)
    # Average across RGB channels
    sv = sv.mean(axis=0)                      # (224, 224)
    # Light smoothing only — preserve facial structure
    sv = gaussian_filter(sv, sigma=3)         # was sigma=8, too blurry
    return sv

# ── Overlay heatmap on image ──────────────────────────────────────
def show_heatmap(ax, img_np, sv, title, title_color="white", show_colorbar=False, fig=None):
    vmax = np.percentile(np.abs(sv), 95)  # was 99, use 95 for better contrast
    vmax = max(vmax, 1e-8)

    ax.imshow(img_np)  # show full image clearly
    im = ax.imshow(
        sv,
        cmap   = "RdBu_r",
        alpha  = 0.65,      # was 0.8, slightly more transparent → face visible
        vmin   = -vmax,
        vmax   =  vmax
    )
    if show_colorbar and fig is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=8, color=title_color, fontweight="bold")
    ax.axis("off")
    return im

# ── Visualize single sample ───────────────────────────────────────
def visualize_shap(
    shap_values,
    img_tensor,
    img_pil,
    true_label:  int,
    pred_label:  int,
    pred_probs:  np.ndarray,
    sample_idx:  int,
    save_path:   Optional[Path] = None,
):
    true_label = int(true_label)
    pred_label = int(pred_label)
    correct    = pred_label == true_label

    img_np = denormalize(img_tensor)

    # ── Layout: 2 rows ──
    # Row 1: Original | Predicted SHAP | True class SHAP | Difference map
    # Row 2: All 7 class heatmaps

    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor('#0f0f0f')

    gs  = gridspec.GridSpec(
        2, 4,
        figure     = fig,
        hspace     = 0.4,
        wspace     = 0.3,
        top        = 0.88,
        bottom     = 0.12
    )

    # ── Row 1, Col 1: Original image ──
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_np)
    ax_orig.set_title(
        f"Original\nTrue: {UNIFIED_CLASSES[true_label]}",
        fontsize=11, fontweight="bold", color="white"
    )
    ax_orig.axis("off")

    # ── Row 1, Col 2: Predicted class SHAP ──
    ax_pred = fig.add_subplot(gs[0, 1])
    sv_pred = get_heatmap(shap_values, pred_label, sample_idx)
    im = show_heatmap(
        ax_pred, img_np, sv_pred,
        title=f"SHAP: {UNIFIED_CLASSES[pred_label]}\nPred ({pred_probs[pred_label]:.1%})",
        title_color="#00ff88" if correct else "#ff4444",
        show_colorbar=True,
        fig=fig
    )

    # ── Row 1, Col 3: True class SHAP (even if wrong prediction) ──
    ax_true = fig.add_subplot(gs[0, 2])
    sv_true = get_heatmap(shap_values, true_label, sample_idx)
    show_heatmap(
        ax_true, img_np, sv_true,
        title=f"SHAP: {UNIFIED_CLASSES[true_label]}\nTrue Class ({pred_probs[true_label]:.1%})",
        title_color="#88aaff",
        show_colorbar=False
    )

    # ── Row 1, Col 4: Absolute importance map ──
    # Shows WHICH pixels matter most regardless of direction
    ax_abs = fig.add_subplot(gs[0, 3])
    sv_abs = np.abs(sv_pred)
    sv_abs = (sv_abs - sv_abs.min()) / (sv_abs.max() - sv_abs.min() + 1e-8)
    ax_abs.imshow(img_np, alpha=0.4)
    ax_abs.imshow(sv_abs, cmap="hot", alpha=0.7)
    ax_abs.set_title(
        f"Pixel Importance\n(Absolute SHAP)",
        fontsize=9, color="white", fontweight="bold"
    )
    ax_abs.axis("off")

    # ── Row 2: All 7 class heatmaps ──
    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, NUM_CLASSES, subplot_spec=gs[1, :],
        wspace=0.15
    )

    for class_idx in range(NUM_CLASSES):
        ax    = fig.add_subplot(gs2[class_idx])
        sv    = get_heatmap(shap_values, class_idx, sample_idx)

        is_pred  = class_idx == pred_label
        is_true  = class_idx == true_label
        if is_true and is_pred:
            t_color = "#00ff88"   # correct prediction
        elif is_true:
            t_color = "#88aaff"   # true class (missed)
        elif is_pred:
            t_color = "#ff4444"   # wrong prediction
        else:
            t_color = "#aaaaaa"   # other class

        show_heatmap(
            ax, img_np, sv,
            title=f"{UNIFIED_CLASSES[class_idx]}\n{pred_probs[class_idx]:.1%}",
            title_color=t_color
        )

        # Border highlight for pred/true classes
        if is_pred or is_true:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(t_color)
                spine.set_linewidth(2)

    # ── Title ──
    status = "✓ Correct" if correct else "✗ Wrong"
    fig.suptitle(
        f"SHAP Explanation  |  "
        f"True: {UNIFIED_CLASSES[true_label]}  |  "
        f"Pred: {UNIFIED_CLASSES[pred_label]}  |  {status}",
        fontsize=13, fontweight="bold",
        color="white", y=0.97
    )

    # ── Legend ──
    legend_text = (
        "🔴 Red = pushes toward predicted class   "
        "🔵 Blue = pushes away   "
        "🟠 Hot = high absolute importance"
    )
    fig.text(
        0.5, 0.03, legend_text,
        ha="center", fontsize=8,
        color="#aaaaaa",
        style="italic"
    )

    if save_path:
        plt.savefig(
            save_path, dpi=150,
            bbox_inches="tight",
            facecolor="#0f0f0f"
        )
        plt.close()
    else:
        plt.show()

    return fig

# ── Main explain function ─────────────────────────────────────────
def explain_model(
    model_name:   str,
    dataset_name: str = "fer2013",
    num_samples:  int = SHAP_TEST_SAMPLES,
    bg_samples:   int = BACKGROUND_SAMPLES,
):
    print(f"\n{'='*60}")
    print(f"  SHAP Explanations: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return None

    # Load model
    model = get_model(model_name, pretrained=False).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()
    model = disable_inplace(model)

    # Load data
    print("\n  Loading background samples...")
    background = load_background_samples(dataset_name, bg_samples)

    print("\n  Loading test samples...")
    images, tensors, labels = load_test_samples(dataset_name, num_samples)

    # Get predictions
    print("\n  Getting predictions...")
    with torch.no_grad():
        logits = model(tensors.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
    pred_labels = np.argmax(probs, axis=1)

    # Compute SHAP
    shap_values = generate_shap_values(model, background, tensors, device)

    # Output directory
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
            f"{idx:03d}_{UNIFIED_CLASSES[true_label]}_" \
            f"pred{UNIFIED_CLASSES[pred_label]}.png"

        visualize_shap(
            shap_values,
            tensors[idx],
            images[idx],
            true_label,
            pred_label,
            probs[idx],
            sample_idx = idx,
            save_path  = save_path
        )

        summary.append({
            "idx":        idx,
            "true":       UNIFIED_CLASSES[true_label],
            "pred":       UNIFIED_CLASSES[pred_label],
            "confidence": float(probs[idx][pred_label]),
            "correct":    correct,
            "saved":      str(save_path)
        })

    # Save summary
    summary_path = out_dir / "shap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    correct_count = sum(s["correct"] for s in summary)
    print(f"\n  Done! {correct_count}/{len(summary)} correct")
    print(f"  Summary → {summary_path}")
    print(f"  Images  → {out_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--dataset",    type=str, default="fer2013")
    parser.add_argument("--samples",    type=int, default=20)
    parser.add_argument("--bg-samples", type=int, default=100)
    parser.add_argument("--all",        action="store_true")
    args = parser.parse_args()

    if args.all:
        from Config import MODELS_TO_TRAIN
        for model_name in MODELS_TO_TRAIN:
            explain_model(model_name, args.dataset, args.samples, args.bg_samples)
    else:
        explain_model(args.model, args.dataset, args.samples, args.bg_samples)