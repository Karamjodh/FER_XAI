import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import shap
from scipy.ndimage import gaussian_filter
from Config import (CKPT_DIR, PLOTS_DIR, UNIFIED_CLASSES, NUM_CLASSES, IMG_SIZE)
from Datasets import get_dataloaders, get_transforms
from Models import get_model, load_checkpoint
BACKGROUND_SAMPLES = 100
SHAP_TEST_SAMPLES = 20
EXPLANATIONS_DIR = Path("outputs/explanations/shap")
EXPLANATIONS_DIR.mkdir(parents = True, exist_ok = True)

def load_background_samples(dataset_name : str, num_samples : int = BACKGROUND_SAMPLES):
    import pandas as pd
    from Config import FER_DIR
    csv_path = FER_DIR / "fer2013.csv"
    df = pd.read_csv(csv_path)
    train_df = df[df["Usage"] == 'Training'].reset_index(drop = True)
    selected = train_df.sample(num_samples, random_state = 42)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    tensors = []
    for _, row in selected.iterrows():
        pixels = np.array(
            row['pixels'].split(), dtype = np.uint8
        ).reshape(48,48)
        img = Image.fromarray(pixels, mode = "L").convert("RGB")
        tensor = transform(img)
        tensors.append(tensor)
    background = torch.stack(tensors)
    print(f"Background samples : {background.shape}")
    return background

def load_test_samples(dataset_name : str, num_samples : int = SHAP_TEST_SAMPLES):
    import pandas as pd
    from Config import FER_DIR
    csv_path = FER_DIR / "fer2013.csv"
    df = pd.read_csv(csv_path)
    test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop = True)
    samples_per_class = max(1, num_samples // NUM_CLASSES)
    selected = []
    for class_idx in range(NUM_CLASSES):
        class_rows = test_df[test_df['emotion'] == class_idx]
        n = min(samples_per_class, len(class_rows))
        selected.append(class_rows.sample(n, random_state = 42))
    selected_df = pd.concat(selected).reset_index(drop = True)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    resize = transforms.Resize((IMG_SIZE,IMG_SIZE))
    images = []
    tensors = []
    labels = []
    for _, row in selected_df.iterrows():
        pixels = np.array(
            row['pixels'].split(), dtype = np.uint8
        ).reshape(48,48)
        img = Image.fromarray(pixels, mode = "L").convert("RGB")
        img = resize(img)
        images.append(img)
        tensors.append(transform(img))
        labels.append(int(row['emotion']))
    tensors = torch.stack(tensors)
    print(f"Test samples loaded : {tensors.shape}")
    return images, tensors, labels

def disable_inplace(model):
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return model

def generate_shap_values(model, background, test_tensors, device):
    print("\n  Computing SHAP values...")

    background   = background.to(device)
    test_tensors = test_tensors.to(device)

    # Use GradientExplainer — works with ResNet residual connections
    explainer = shap.GradientExplainer(model, background)

    shap_values_list = []
    for i in range(len(test_tensors)):
        print(f"  Computing SHAP [{i+1}/{len(test_tensors)}]...")
        sv = explainer.shap_values(test_tensors[i:i+1])
        shap_values_list.append(sv)

    # Debug — check what shape we actually got
    print(f"  Type of shap_values_list[0]: {type(shap_values_list[0])}")
    if isinstance(shap_values_list[0], list):
        print(f"  Length: {len(shap_values_list[0])}")
        print(f"  Shape of [0][0]: {shap_values_list[0][0].shape}")
    else:
        print(f"  Shape: {shap_values_list[0].shape}")

    # Reorganize correctly
    # Shape is (1, 3, 224, 224, 7) — last dim is classes
    # Stack all samples → (N, 3, 224, 224, 7)
    stacked = np.concatenate(shap_values_list, axis=0)
    print(f"  Stacked shape: {stacked.shape}")

    # Split into list of 7 arrays each (N, 3, 224, 224)
    shap_values = [stacked[..., c] for c in range(NUM_CLASSES)]

    print(f"  SHAP values computed!")
    print(f"  Num classes: {len(shap_values)}")
    print(f"  Shape per class: {shap_values[0].shape}")
    return shap_values

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

    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * std + mean, 0, 1)

    def get_heatmap(class_idx):
        sv = shap_values[class_idx][sample_idx]  # (3, 224, 224)
        sv = sv.mean(axis=0)                      # (224, 224)
        sv = gaussian_filter(sv, sigma=8)         # smooth
        return sv

    def show_heatmap(ax, sv, title, title_color="black"):
        vmax = np.percentile(np.abs(sv), 99)      # robust scaling
        vmax = max(vmax, 1e-6)
        ax.imshow(img_np, alpha=0.3)
        ax.imshow(sv, cmap="RdBu_r",
                  alpha=0.8, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8,
                     color=title_color, fontweight="bold")
        ax.axis("off")

    fig = plt.figure(figsize=(20, 10))

    # Original
    ax_orig = fig.add_subplot(2, 4, 1)
    ax_orig.imshow(img_np)
    ax_orig.set_title(
        f"Original\nTrue: {UNIFIED_CLASSES[true_label]}",
        fontsize=11, fontweight="bold"
    )
    ax_orig.axis("off")

    # Predicted class SHAP
    ax_pred = fig.add_subplot(2, 4, 2)
    sv_pred = get_heatmap(pred_label)
    vmax    = np.percentile(np.abs(sv_pred), 99)
    vmax    = max(vmax, 1e-6)
    ax_pred.imshow(img_np, alpha=0.3)
    im = ax_pred.imshow(sv_pred, cmap="RdBu_r",
                        alpha=0.8, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax_pred, fraction=0.046)
    ax_pred.set_title(
        f"SHAP: {UNIFIED_CLASSES[pred_label]}\n"
        f"Pred ({pred_probs[pred_label]:.1%})",
        fontsize=11, fontweight="bold",
        color="green" if pred_label == true_label else "red"
    )
    ax_pred.axis("off")

    # All 7 classes bottom row
    for class_idx in range(NUM_CLASSES):
        ax    = fig.add_subplot(2, 7, 8 + class_idx)
        sv    = get_heatmap(class_idx)
        color = "green" if class_idx == true_label else "black"
        show_heatmap(
            ax, sv,
            f"{UNIFIED_CLASSES[class_idx]}\n"
            f"{pred_probs[class_idx]:.1%}",
            title_color=color
        )

    fig.suptitle(
        f"SHAP Explanation | "
        f"True: {UNIFIED_CLASSES[true_label]} | "
        f"Pred: {UNIFIED_CLASSES[pred_label]} "
        f"({'Correct' if pred_label == true_label else 'Wrong'})",
        fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return fig
    

# ================================================================
# MAIN EXPLAIN FUNCTION
# ================================================================
# Ties everything together:
#   1. Loads model checkpoint
#   2. Loads background samples (from training set)
#   3. Loads test samples
#   4. Computes SHAP values for ALL test samples at once
#      (more efficient than one by one)
#   5. Visualizes and saves each explanation
# ================================================================

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

    # Check checkpoint exists
    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return None

    # Load model
    model = get_model(model_name, pretrained=False).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()
    model = disable_inplace(model)
    # Load background + test samples
    print("\n  Loading background samples...")
    background = load_background_samples(dataset_name, bg_samples)

    print("\n  Loading test samples...")
    images, tensors, labels = load_test_samples(dataset_name, num_samples)

    # Get model predictions for all test samples
    print("\n  Getting model predictions...")
    with torch.no_grad():
        logits = model(tensors.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()

    pred_labels = np.argmax(probs, axis=1)

    # Compute SHAP values for ALL samples at once
    # More efficient than computing one by one
    shap_values = generate_shap_values(
        model, background, tensors, device
    )

    # Output directory
    out_dir = EXPLANATIONS_DIR / f"{model_name}_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    # Visualize each sample
    print("\n  Generating visualizations...")
    for idx in range(len(images)):
        true_label = int(labels[idx])
        pred_label = int(pred_labels[idx])
        correct    = pred_label == true_label

        print(f"  [{idx+1}/{len(images)}] "
              f"True: {UNIFIED_CLASSES[true_label]} | "
              f"Pred: {UNIFIED_CLASSES[pred_label]} "
              f"({'✅' if correct else '❌'})")

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


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--dataset", type=str, default="fer2013")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of test images to explain")
    parser.add_argument("--bg-samples", type=int, default=100,
                        help="Number of background samples")
    parser.add_argument("--all", action="store_true",
                        help="Explain all 3 models")
    args = parser.parse_args()

    if args.all:
        from Config import MODELS_TO_TRAIN
        for model_name in MODELS_TO_TRAIN:
            explain_model(
                model_name,
                args.dataset,
                args.samples,
                args.bg_samples
            )
    else:
        explain_model(
            args.model,
            args.dataset,
            args.samples,
            args.bg_samples
        )