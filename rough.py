# import torch
# import torch.nn as nn
# from torch.cuda.amp import GradScaler
# from Models import get_model
# from Datasets import get_dataloaders
# from Train import run_epoch, build_optimizer, set_seed

# if __name__ == '__main__':
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load data
#     train_loader, val_loader, _, class_weights = get_dataloaders("fer2013")

#     # Build model
#     model = get_model("resnet50").to(device)
#     model.freeze_backbone()

#     # Loss + optimizer + scaler
#     criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
#     optimizer = build_optimizer(model, lr=1e-3)
#     scaler    = GradScaler()

#     # Run ONE train batch only to test
#     # We temporarily shrink loader to 1 batch
#     print("\nTesting one train epoch (1 batch only)...")
#     model.train()
#     images, labels = next(iter(train_loader))
#     images, labels = images.to(device), labels.to(device)

#     optimizer.zero_grad(set_to_none=True)
#     outputs = model(images)
#     loss    = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     preds   = outputs.argmax(dim=1)
#     acc     = (preds == labels).float().mean()
#     print(f"  Loss: {loss.item():.4f}")
#     print(f"  Acc:  {acc.item():.4f}")
#     print(f"  Output shape: {outputs.shape}")
#     print("\n✅ Single batch forward+backward pass working!")

from Train import train_model

if __name__ == '__main__':
    # Temporarily set small epochs in config.py for testing:
    # PHASE1_EPOCHS = 1
    # PHASE2_EPOCHS = 1
    train_model("resnet50", "fer2013")