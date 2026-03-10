import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import label_binarize
from Config import (
    CKPT_DIR, PLOTS_DIR, UNIFIED_CLASSES, NUM_CLASSES, MODELS_TO_TRAIN
)
from Models import get_model, load_checkpoint
from Datasets import get_dataloaders

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim = 1)
            preds = logits.argmax(dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

def plot_confusion_matrix(labels, preds, model_name: str,
                          dataset: str):
    cm = confusion_matrix(labels, preds)

    # Normalize — convert counts to percentages
    # So each row sums to 1.0
    # Easier to compare across classes with different sample sizes
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=UNIFIED_CLASSES,
        yticklabels=UNIFIED_CLASSES,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True", fontsize=12)
    axes[0].set_title("Confusion Matrix (Raw Counts)",
                      fontsize=13, fontweight="bold")

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=UNIFIED_CLASSES,
        yticklabels=UNIFIED_CLASSES,
        ax=axes[1], linewidths=0.5
    )
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True", fontsize=12)
    axes[1].set_title("Confusion Matrix (Normalized)",
                      fontsize=13, fontweight="bold")

    fig.suptitle(f"{model_name} — Confusion Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = PLOTS_DIR / f"{model_name}_{dataset}_confusion_matrix.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Confusion matrix saved → {fname}")
    return fname

def plot_roc_curves(labels, probs, model_name: str,
                   dataset: str):
    # Binarize labels for one-vs-rest ROC
    # [3, 0, 6] → [[0,0,0,1,0,0,0], [1,0,0,0,0,0,0], ...]
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    auc_scores = {}
    for i, (cls_name, color) in enumerate(zip(UNIFIED_CLASSES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls_name] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1.5,
            label="Random Classifier (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} — ROC Curves (One-vs-Rest)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fname = PLOTS_DIR / f"{model_name}_{dataset}_roc_curves.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 ROC curves saved → {fname}")
    return auc_scores

def plot_training_curves(history: dict, model_name: str,
                         dataset: str):
    epochs     = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = history["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"],
             label="Train Loss", color="#2196F3", lw=2)
    ax1.plot(epochs, history["val_loss"],
             label="Val Loss", color="#F44336", lw=2)
    ax1.axvline(best_epoch, color="green", ls="--",
                alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"],
             label="Train Acc", color="#2196F3", lw=2)
    ax2.plot(epochs, history["val_acc"],
             label="Val Acc", color="#F44336", lw=2)
    ax2.axhline(history["best_val_acc"], color="green",
                ls="--", alpha=0.7,
                label=f"Best Val Acc ({history['best_val_acc']:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{model_name} — Training History",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = PLOTS_DIR / f"{model_name}_{dataset}_training_curves.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Training curves saved → {fname}")
    return fname

def plot_model_comparison(results: dict, dataset: str):
    model_names = list(results.keys()) + ["AlexNet\n(Base Paper)"]
    metrics     = ["accuracy", "macro_f1", "weighted_f1", "mean_auc"]
    labels      = ["Accuracy", "Macro F1", "Weighted F1", "Mean AUC"]
    colors      = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    # Add AlexNet baseline values
    alexnet_baseline = {
        "accuracy":    0.65,
        "macro_f1":    0.61,
        "weighted_f1": 0.63,
        "mean_auc":    0.72,
    }
    all_results = dict(results)
    all_results["AlexNet\n(Base Paper)"] = alexnet_baseline

    x     = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, label, color) in enumerate(
        zip(metrics, labels, colors)
    ):
        values = [all_results[m].get(metric, 0)
                  for m in model_names]
        bars   = ax.bar(x + i * width, values, width,
                        label=label, color=color, alpha=0.85)

        # Add value labels on top of bars
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Comparison — {dataset.upper()}\n"
        f"Our Models vs AlexNet Baseline (Base Paper)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Highlight our improvement
    ax.axhline(0.65, color="red", ls="--",
               alpha=0.5, lw=1.5)

    plt.tight_layout()

    fname = PLOTS_DIR / f"model_comparison_{dataset}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Comparison chart saved → {fname}")
    return fname

def evaluate_model(model_name : str, dataset_name : str = "fer2013"):
    print(f"\n{'='*60}")
    print(f"Evaluating : {model_name.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"
    if not ckpt_path.exists():
        print(f"Checkpoint not found : {ckpt_path}")
        print(f"Run : python Train.py --model {model_name} first")
        return None
    model = get_model(model_name,pretrained = False).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    _, _, test_loader, _ = get_dataloaders(dataset_name)
    print("\n Running model on test set...")
    preds, labels, probs = get_predictions(model, test_loader, device)
    report = classification_report(labels, preds, target_names=UNIFIED_CLASSES, output_dict=True)
    print("\n Classification Report : ")
    print(classification_report(labels, preds, target_names=UNIFIED_CLASSES))
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels,preds,average = "macro"))
    weighted_f1 = float(f1_score(labels, preds, average = "weighted"))
    macro_prec = float(precision_score(labels, preds, average = "macro"))
    macro_rec = float(recall_score(labels, preds, average = "macro"))
    print(f" Accuracy : {accuracy :.4f}")
    print(f" Macro F1 : {macro_f1 :.4f}")
    print(f" Weighted F1 : {weighted_f1 :.4f}")
    print(f" AcMacro Precision : {macro_prec :.4f}")
    print(f" Macro Recall : {macro_rec :.4f}")
    print(f"\n Generating plots...")
    plot_confusion_matrix(labels, preds, model_name, dataset_name)
    auc_scores = plot_roc_curves(labels,probs,model_name,dataset_name)
    mean_auc = float(np.mean(list(auc_scores.values())))
    print(f"Mean AUC : {mean_auc :.4f}")
    hist_path = CKPT_DIR / f"{model_name}_{dataset_name}_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_curves(history, model_name, dataset_name)
    results = {
        "model" : model_name,
        "dataset" : dataset_name,
        "accuracy" : accuracy,
        "macro_f1" : macro_f1,
        "weighted_f1" : weighted_f1,
        "macro_precision" : macro_prec,
        "macro_recall" : macro_rec,
        "mean_auc" : mean_auc,
        "per_class_auc" : {k : float(v) for k,v in auc_scores.items()},
        "classification_report" : report
    }
    res_path = CKPT_DIR / f"{model_name}_{dataset_name}_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent = 2)
    print(f"\n Results Saved → {res_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "resnet50", choices = ["resnet50","efficientnet_b0"])
    parser.add_argument("--dataset", type = str, default = "fer2013", choices = ["fer2013","rafdb","combined"])
    parser.add_argument("--all", action = "store_true", help = "Evaluate all 3 model  and compare")
    args = parser.parse_args()
    if args.all:
        all_results = {}
        for model_name in MODELS_TO_TRAIN:
            r = evaluate_model(model_name, args.dataset)
            if r:
                all_results[model_name] = r
        if len(all_results) > 1 :
            plot_model_comparison(all_results,args.dataset)
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"{'Model' : <22} {'Accuracy' : > 10} {'Macro F1' :> 10} {'Mean AUC' :> 10}")
        print(f"{'='*55}")
        for name, r in all_results.items():
            print(f"{name :<22} {r['accuracy'] :> 10.4f}"
                  f"{r['macro_f1'] :> 10.4f} {r['mean_auc'] :> 10.4f}")
        print(f"{'AlexNet (Baseline)' : 22}"
              f"{'~0.6500' : >10} {'~0.6100' : >10} {'~0.7200' : >10}")
    else:
        evaluate_model(args.model, args.dataset)