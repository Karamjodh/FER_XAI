import json
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from Config import (
PHASE1_EPOCHS,PHASE1_LR,PHASE2_EPOCHS,PHASE2_LR,PHASE2_UNFREEZE,OPTIMIZER,WEIGHT_DECAY,SCHEDULER1,LR_STEP_SIZE,LR_GAMMA,LR_MIN,SCHEDULER2,CYCLIC_BASE_LR,CYCLIC_MAX_LR,CYCLIC_STEP,ACTIVE_SCHEDULER,USE_WEIGHTED_LOSS,USE_AMP,LOG_INTERVAL,CKPT_DIR,SEED,EARLY_STOP_PATIENCE,MODELS_TO_TRAIN
)
from Datasets import get_dataloaders
from Models import get_model, save_checkpoint

def set_seed(seed : int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience : int = EARLY_STOP_PATIENCE,min_delta : float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss : float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1 
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
def build_optimizer(model, lr:float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER == "adam":
        return torch.optim.Adam(trainable, lr = lr, weight_decay = WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        return torch.optim.AdamW(trainable, lr = lr, weight_decay = WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

def build_scheduler(optimizer, phase_epochs : int):
    if ACTIVE_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = phase_epochs, eta_min = LR_MIN)
    elif ACTIVE_SCHEDULER == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = CYCLIC_BASE_LR, max_lr = CYCLIC_MAX_LR, step_size_up = CYCLIC_STEP, mode = "triangular2", cycle_momentum = False)
    elif ACTIVE_SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer,step_size = LR_STEP_SIZE,gamma = LR_GAMMA)
    return None

def run_epoch(model, loader, criterion, optimizer, scaler, device, phase : str = "train"):
    is_train = (phase == "train")
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            if is_train:
                optimizer.zero_grad(set_to_none = True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs,labels)
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm = 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            preds = outputs.argmax(dim = 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            if is_train and (batch_idx + 1) % LOG_INTERVAL == 0:
                running_acc = correct/total
                print(f"Batch[{batch_idx+1:4d}/{len(loader)} ]"
                      f' Loss : {loss.item():.4f} |'
                      f' Acc : {running_acc:.4f}')
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model_name : str, dataset_name: str = "fer2013"):
    print(f"\n{('='*60)}")
    print(f"Training : {model_name.upper()} on {dataset_name.upper()}")
    print(f"{('='*60)}")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    train_loader, val_loader, _,class_weights = get_dataloaders(dataset_name)
    model = get_model(model_name).to(device)
    if USE_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    history = {
        'model' : model_name,
        "dataset" : dataset_name,
        "train_loss" : [],
        "train_acc" : [],
        "val_loss" : [],
        "val_acc" : [],
        "best_val_loss" : 0.0,
        "best_epoch" : 0,
    }
    best_val_acc = 0.0
    ckpt_path = CKPT_DIR / f"{model_name}_{dataset_name}_best.pth"

    print(f"\n PHASE 1 - Head only | {PHASE1_EPOCHS} epochs | LR={PHASE1_LR}")
    model.freeze_backbone()
    optimizer = build_optimizer(model,PHASE1_LR)
    scheduler = build_scheduler(optimizer,PHASE1_EPOCHS)
    early_stop = EarlyStopping(patience=5)
    for epoch in range(1,PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion,optimizer,scaler,device,"train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion,optimizer,scaler,device,"eval")
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[P1 Epoch {epoch : 2d}/{PHASE1_EPOCHS}]"
              f'Train Loss : {train_loss :.4f} Acc : {train_acc :.4f} |'
              f"Val Loss : {val_loss :.4f} Acc : {val_acc :.4f}"
              f"Time : {elapsed : .1f}s")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model,optimizer,epoch,val_acc, str(ckpt_path))
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch

        if early_stop.step(val_loss):
            print(f"Early Stopping at epoch {epoch}")
            break

    print(f"\n PHASE 2 - Fine-tuning | {PHASE2_EPOCHS} epochs | LR={PHASE2_LR}")
    model.unfreeze_last_n(PHASE2_UNFREEZE)
    optimizer = build_optimizer(model,PHASE2_LR)
    scheduler = build_scheduler(optimizer,PHASE2_EPOCHS)
    ealy_stop = EarlyStopping(patience = EARLY_STOP_PATIENCE)
    
    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model,train_loader,criterion,optimizer,scaler,device,"train")
        val_loss, val_acc = run_epoch(model,val_loader,criterion,optimizer,scaler,device,"eval")
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[P2 Epoch {epoch : 2d}/{PHASE2_EPOCHS}]"
              f'Train Loss : {train_loss :.4f} Acc : {train_acc :.4f} |'
              f"Val Loss : {val_loss :.4f} Acc : {val_acc :.4f}"
              f"Time : {elapsed : .1f}s")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model,optimizer,epoch,val_acc, str(ckpt_path))
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = PHASE1_EPOCHS + epoch
        if early_stop.step(val_loss):
            print(f"Early Stopping at epoch {epoch}")
            break
    print(f"\n Best Val Acc : {best_val_acc:.4f} "
          f"(Epoch {history['best_epoch']})")
    hist_path = CKPT_DIR / f"{model_name}_{dataset_name}_history.json"
    with open(hist_path,'w') as f:
        json.dump(history,f,indent = 2)
    print(f"History Saved → {hist_path}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type = str, default = "resnet50", choices = ["resnet50","efficientnet_b0"])
    parser.add_argument("--dataset",type = str, default = "fer2013", choices = ["fer2013","rafdb","combined"])
    parser.add_argument("--all",action = "store_true", help = "Train all 3 models sequentially")
    args = parser.parse_args()
    if args.all:
        for model_name in MODELS_TO_TRAIN:
            train_model(model_name,args.dataset)
    else:
        train_model(args.model,args.dataset)
