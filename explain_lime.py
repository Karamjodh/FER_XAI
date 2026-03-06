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
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from Config import (
    CKPT_DIR, PLOTS_DIR, UNIFIED_CLASSES, NUM_CLASSES, IMG_SIZE
)
from Datasets import get_dataloaders, get_transforms
from Models import get_model, load_checkpoint

NUM_SAMPLES = 1000
NUM_FEATURES = 10
BATCH_SIZE_LIME = 32
EXPLANATIONS_DIR = Path("outputs/explanations/lime")
EXPLANATIONS_DIR.mkdir(parents = True, exist_ok = True)

def get_lime_transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                   mean = [0.485, 0.456, 0.406],
                                   std = [0.229, 0.224, 0.225]
                               )])

def make_predict_fn(model,device):
    transform = get_lime_transform()
    def predict_fn(images : np.ndarray) -> np.ndarray:
        batch = torch.stack([
            transform(Image.fromarray(img.astype(np.uint8))) for img in images]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim = 1)
        return probs.cpu().numpy()
    return predict_fn

def load_test_samples(dataset_name : str, num_samples: int = 20):
    from Datasets import FER2013Dataset
    import pandas as pd
    from Config import FER_DIR
    csv_path = FER_DIR / "fer2013.csv"
    df = pd.read_csv(csv_path)
    test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop = True)
    if num_samples < NUM_CLASSES:
        selected_df = test_df.sample(num_samples, random_state = 42)
    else:
        samples_per_class = num_samples // NUM_CLASSES
        selected = []
        for class_idx in range(NUM_CLASSES):
            class_rows = test_df[test_df["emotion"] == class_idx]
            n = min(samples_per_class, len(class_rows))
            selected.append(class_rows.sample(n, random_state = 42))
        selected_df = pd.concat(selected).reset_index(drop = True)
    images = []
    labels = []
    resize = transforms.Resize((IMG_SIZE,IMG_SIZE))
    for _, row in selected_df.iterrows():
        pixels = np.array(row["pixels"].split(), dtype = np.uint8).reshape(48,48)
        img = Image.fromarray(pixels, mode = "L").convert("RGB")
        img = resize(img)
        images.append(img)
        labels.append(int(row["emotion"]))
    print(f"Loaded {len(images)} test samples ")
    return images, labels

def generate_lime_explanation(image : Image.Image, predict_fn, num_samples : int = NUM_SAMPLES):
    explainer = lime_image.LimeImageExplainer()
    img_array = np.array(image)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size = 4, max_dist = 200, ratio = 0.2)
    explanation = explainer.explain_instance(
        img_array, predict_fn, top_labels = NUM_CLASSES,hide_color = 0, num_samples = num_samples, segmentation_fn = segmenter
    )
    return explanation, img_array 

def visualize_explanation(
    explanation,
    img_array:    np.ndarray,
    true_label:   int,
    pred_label:   int,
    pred_probs:   np.ndarray,
    save_path:    Optional[Path] = None,
    num_features: int = NUM_FEATURES,
):
    # Convert to native Python int to match LIME's keys
    pred_label = int(pred_label)
    true_label = int(true_label)

    # Check available labels in explanation
    available_labels = list(explanation.local_exp.keys())
    if pred_label not in available_labels:
        print(f"  Warning: pred_label {pred_label} not in "
              f"explanation labels {available_labels}")
        pred_label = available_labels[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: Original Image ──
    axes[0].imshow(img_array)
    axes[0].set_title(
        f"Original\nTrue: {UNIFIED_CLASSES[true_label]}",
        fontsize=12, fontweight="bold"
    )
    axes[0].axis("off")

    # ── Panel 2: Positive Regions Only ──
    temp, mask = explanation.get_image_and_mask(
        pred_label,
        positive_only = True,
        num_features  = num_features,
        hide_rest     = False
    )
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title(
        f"Supporting Regions\nPred: {UNIFIED_CLASSES[pred_label]} "
        f"({pred_probs[pred_label]:.1%})",
        fontsize=12, fontweight="bold",
        color="green" if pred_label == true_label else "red"
    )
    axes[1].axis("off")

    # ── Panel 3: Positive + Negative Regions ──
    temp, mask = explanation.get_image_and_mask(
        pred_label,
        positive_only = False,
        num_features  = num_features,
        hide_rest     = False
    )
    axes[2].imshow(mark_boundaries(temp, mask))
    axes[2].set_title(
        f"All Regions\nGreen=For | Red=Against",
        fontsize=12
    )
    axes[2].axis("off")

    # ── Probability bar at bottom ──
    fig.suptitle(
        " | ".join([
            f"{UNIFIED_CLASSES[i]}: {pred_probs[i]:.1%}"
            for i in range(NUM_CLASSES)
        ]),
        fontsize=9, y=0.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig

def explain_model(
        model_name : str,
        dataset_name : str = "fer2013",
        num_samples : int = 20,
        lime_samples : int = NUM_SAMPLES,
):
    print(f"\n{'='*60}")
    print(f"LIME EXPLANATION : {model_name} on {dataset_name}")
    print(f"\n{'='*60}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"
    model = get_model(model_name, pretrained = False).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()
    predict_fn = make_predict_fn(model,device)
    print(' Loading test samples...')
    images, labels = load_test_samples(dataset_name, num_samples)
    out_dir = EXPLANATIONS_DIR / f"{model_name}_{dataset_name}"
    out_dir.mkdir(parents = True, exist_ok = True)
    summary = []
    for idx, (image, true_label) in enumerate(zip(images, labels)):
        print(f"\n [{idx+1}/{len(images)}]"
              f"True : {UNIFIED_CLASSES[true_label]}")
        transform = get_lime_transform()
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim = 1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs).item())
        correct = pred_label == true_label
        print(f"Pred : {UNIFIED_CLASSES[pred_label]}"
              f'({probs[pred_label]:.1%})'
              f"{'COMPLETED' if correct else 'NOT-COMPLETED'}")
        print(f"Generating LIME Explanation "
              f"({lime_samples} samples)...")
        explanation, img_array = generate_lime_explanation(image, predict_fn, lime_samples)
        save_path = out_dir/ \
            f"{idx:03d}_{UNIFIED_CLASSES[true_label]}_"\
            f"pred{UNIFIED_CLASSES[pred_label]}.png"
        visualize_explanation(explanation, img_array, int(true_label), int(pred_label),probs, save_path = save_path)
        print(f"Saved → {save_path.name}")
        summary.append({
            "idx" : idx,
            "true" : UNIFIED_CLASSES[true_label],
            "pred" : UNIFIED_CLASSES[pred_label],
            "confidence" : float(probs[pred_label]),
            "correct" : correct,
            "saved" : str(save_path)
        })
    summary_path = out_dir / "lime_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent = 2)
    correct_count = sum(s["correct"] for s in summary)
    print(f"\n Done! {correct_count}/{len(summary)} correct")
    print(f"Summary → {summary_path}")
    print(f"Images → {out_dir}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "resnet50", choices = ["resnet50","efficientnet_b0", "vgg16"])
    parser.add_argument("--dataset",type = str, default = "fer2013")
    parser.add_argument("--samples", type = int, default = 20, help = "Number of test images to explain")
    parser.add_argument("--lime-samples", type=int, default=1000,help="Number of LIME perturbations per image")
    args = parser.parse_args()
    explain_model(args.model,args.dataset,args.samples,args.lime_samples)