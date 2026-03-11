import json
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from Config import (
    EPOCHS, LR, LR_MIN, OPTIMIZER, WEIGHT_DECAY,
    USE_AMP, LOG_INTERVAL, CKPT_DIR, SEED,
    EARLY_STOP_PATIENCE, MODELS_TO_TRAIN, LABEL_SMOOTHING,
    USE_WEIGHTED_LOSS
)
from Datasets import get_dataloaders
from Models import get_model, save_checkpoint


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adam":
        return torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")


def run_epoch(model, loader, criterion, optimizer, scaler, device, phase: str = "train"):
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            if is_train and (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"  Batch [{batch_idx+1:4d}/{len(loader)}]"
                      f"  Loss: {loss.item():.4f}"
                      f"  Acc: {correct/total:.4f}")

    return total_loss / total, correct / total


def train_model(model_name: str, dataset_name: str = "fer2013"):
    print(f"\n{'='*60}")
    print(f"Training : {model_name.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _, class_weights = get_dataloaders(dataset_name)

    # ── Model ──────────────────────────────────────────────────────────────
    model = get_model(model_name).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    # label_smoothing improves accuracy 1-3% on noisy datasets like FER2013
    if USE_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=LABEL_SMOOTHING
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    # Single-phase: train ALL layers from epoch 1 with low LR (1e-4)
    # CosineAnnealingLR smoothly decays → better than CyclicLR
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN
    )
    scaler = torch.cuda.amp.GradScaler()
    early_stop  = EarlyStopping()
    best_val_acc = 0.0
    ckpt_path   = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"

    history = {
        "model":        model_name,
        "dataset":      dataset_name,
        "train_loss":   [],
        "train_acc":    [],
        "val_loss":     [],
        "val_acc":      [],
        "best_val_acc": 0.0,
        "best_epoch":   0,
    }

    print(f"\n Single-Phase Training | {EPOCHS} epochs | LR={LR} | "
          f"label_smoothing={LABEL_SMOOTHING} | scheduler=CosineAnnealing\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, "train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, "eval"
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch:2d}/{EPOCHS}]"
              f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}"
              f"  |  Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
              f"  |  LR: {current_lr:.2e}  |  {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, str(ckpt_path))
            history["best_val_acc"] = best_val_acc
            history["best_epoch"]   = epoch
            print(f"  ✔ Saved best model (val_acc={best_val_acc:.4f})")

        if early_stop.step(val_loss):
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\n Best Val Acc : {best_val_acc:.4f}  (Epoch {history['best_epoch']})")

    hist_path = CKPT_DIR / f"{model_name}_{dataset_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {hist_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--dataset", type=str, default="fer2013",
                        choices=["fer2013", "rafdb"])
    parser.add_argument("--all",     action="store_true",
                        help="Train all models sequentially")
    args = parser.parse_args()

    if args.all:
        for model_name in MODELS_TO_TRAIN:
            train_model(model_name, args.dataset)
    else:
        train_model(args.model, args.dataset)