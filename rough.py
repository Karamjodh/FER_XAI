from Datasets import get_dataloaders

if __name__ == '__main__':
    train_loader, val_loader, test_loader, class_weights = get_dataloaders("fer2013")
    imgs, labels = next(iter(train_loader))
    print(f"Images shape : {imgs.shape}")
    print(f"Labels : {labels[:8]}")