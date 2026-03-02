import json
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from Config import (
PHASE1_EPOCHS,PHASE1_LR,PHASE2_EPOCHS,PHASE2_LR,PHASE2_UNFREEZE,OPTIMIZER,WEIGHT_DECAY,SCHEDULER,LR_STEP_SIZE,LR_GAMMA,LR_MIN,USE_WEIGHTED_LOSS,USE_AMP,LOG_INTERVAL,CKPT_DIR,SEED,EARLY_STOP_PATIENCE,MODELS_TO_TRAIN
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
    if SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = phase_epochs, eta_min = LR_MIN)
    elif SCHEDULER == "step":
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
            image = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            if is_train:
                optimizer.zero_grad(set_to_none = True)
            with autocast(enabled = USE_AMP):
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
                print(f"Batch[{batch_idx+1:4d}/{len(loader)}]"
                      f'Loss : {loss.item():.4f} |'
                      f'Acc : {running_acc:.4f}')
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy