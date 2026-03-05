import torch
import torch.nn as nn
from torchvision import models
from Config import NUM_CLASSES, PHASE2_UNFREEZE

class FERModel(nn.Module):
    def __init__(self,backbone : nn.Module, feature_dim : int, model_name : str, dropout : float = 0.5):
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, NUM_CLASSES)

    def forward(self,x):
        features = self.backbone(x)
        features = self.dropout(features)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_n(self, n : int = PHASE2_UNFREEZE):
        for param in self.backbone.parameters():
            param.requires_grad = False
        all_params = list(self.backbone.parameters())
        for param in all_params[-n:]:
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[{self.model_name}] {n} layes unfrozen")
        print(f"Trainable parameters : {trainable:,}/{total:,}")
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def build_resnet50(pretrained : bool = True) -> FERModel:
    backbone = models.resnet50(
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    )
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity() # remove the final classification layer
    model = FERModel(backbone, feature_dim, "ResNet-50",dropout = 0.3)
    print(f"ResNet-50 ready | Feature dim: {feature_dim} | Classes: {NUM_CLASSES}")
    return model
    
def build_efficientnet_b0(pretrained : bool = True) -> FERModel:
    backbone = models.efficientnet_b0(
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity() # remove the final classification layer
    model = FERModel(backbone, feature_dim, "EfficientNet-B0", dropout = 0.4)
    print(f" EfficientNet-B0 ready | Feature dim: {feature_dim} | Classes: {NUM_CLASSES}")
    return model
    
def build_vgg16(pretrained : bool = True) -> FERModel:
    backbone = models.vgg16(
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    )
    backbone.classifier = nn.Sequential(
        nn.Linear(25088,4096),
        nn.ReLU(inplace = True),
        nn.Dropout(0.5),
        nn.Linear(4096,1024),
        nn.ReLU(inplace = True),
    )
    model = FERModel(backbone, 1024, "VGG-16", dropout = 0.5)
    print(f" VGG-16 ready | Feature dim: 1024 | Classes: {NUM_CLASSES}")
    return model
    
MODEL_REGISTRY = {
    "resnet50" : build_resnet50,
    "efficientnet_b0" : build_efficientnet_b0,
    "vgg16" : build_vgg16
}

def get_model(model_name : str, pretrained : bool = True) -> FERModel:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Uknown model'{model_name}'."
            f"Choose from : {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](pretrained=pretrained)

def save_checkpoint(model : FERModel, optimizer, epoch : int,val_acc : float, path: str):
    torch.save({
        "epoch" : epoch,
        "model_name": model.model_name,
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "val_acc" : val_acc
    },path)
    print(f"Saved checkpoint → {path} (val_acc={val_acc:.4f})")

def load_checkpoint(model : FERModel, path : str, device = torch.device, optimizer = None):
    ckpt = torch.load(path,map_location = device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded checkpoint → epoch {ckpt.get('epoch','?')} "f"| val_acc={ckpt.get('val_acc', 0):.4f}")
    return ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0)