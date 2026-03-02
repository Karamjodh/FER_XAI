import torch
from Models import get_model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for name in ["resnet50", "efficientnet_b0", "vgg16"]:
        model = get_model(name).to(device)
        model.freeze_backbone()

        # Dummy forward pass — 2 images, 3 channels, 224x224
        x   = torch.randn(2, 3, 224, 224).to(device)
        out = model(x)

        print(f"  Output shape : {out.shape}")  # should be (2, 7)
        print(f"  Trainable params : {model.get_trainable_params():,}\n")
        
        # Free VRAM between models
        del model
        torch.cuda.empty_cache()