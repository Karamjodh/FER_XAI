import numpy as np
import pandas as pd
from PIL import Image 
from pathlib import Path 
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T # Torch library for image Transformation

from Config import (
    FER_DIR,RAFDB_DIR,IMG_SIZE,BATCH_SIZE,NUM_WORKERS,PIN_MEMORY,NORMALIZE_MEAN,NORMALIZE_STD,AUGMENT_TRAIN,HORIZONTAL_FLIP,RANDOM_ROTATION,COLOR_JITTER,RANDOM_ERASING,UNIFIED_CLASSES,SEED
) # Imported all the necessary parameters from config.py

def get_transforms(split : str) -> T.Compose:
    if split == 'train' and AUGMENT_TRAIN:
        return T.Compose([
            T.Resize((IMG_SIZE,IMG_SIZE)), # resize the images to 224x224
            T.Grayscale(num_output_channels = 3), # converted grayscale to 3 phase RGB
            T.RandomHorizontalFlip(p=0.5), # randomly flip the image horizontally with a probability of 0.5
            T.RandomRotation(degrees=RANDOM_ROTATION), # randomly rotate the image by a random angle between -15 and 15 degrees
            T.ColorJitter(brightness=0.2, contrast = 0.2), # randomly change the brightness and contrast of the image by a factor of 0.2
            T.ToTensor(), # convert the image to a PyTorch tensor and scale pixel values to [0,1]
            T.Normalize(mean = NORMALIZE_MEAN, std = NORMALIZE_STD), # normalized the images by dividing the pixel intensities by 255 and then subtracting the mean and dividing by the standard deviation of the ImageNet dataset
            T.RandomErasing(p=0.2) # randomly erase a rectangular region of the image with a probability of 0.5, which helps the model learn to be more robust to occlusion and missing parts of the image
        ])
    
    return T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.Grayscale(num_output_channels = 3),
        T.ToTensor(),
        T.Normalize(mean = NORMALIZE_MEAN, std = NORMALIZE_STD)
    ])

class FER2013Dataset(Dataset):
    FER_TO_UNIFIED = {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 5,
        5 : 6,
        6 : 4,
    }
    def __init__(self, split : str = "train", transform = None):
        split_map = {
            "train" : "Training",
            "val" : "PublicTest",
            "test" : "PrivateTest"
        }
        csv_path = FER_DIR / "fer2013.csv"
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split_map[split]].reset_index(drop = True) # Created dataframe of the datapoints corresponding to the split i.e train
        self.pixels = df["pixels"].values
        self.labels = df["emotion"].values
        self.transform = transform
        self.split = split
        print(f"FER2013[{split}] : {len(self.labels)} samples")
        print(f"Class distribution : {dict(Counter(self.labels))}")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        pixel_vals = np.array(self.pixels[idx].split(),dtype = np.uint8).reshape(48,48)
        image = Image.fromarray(pixel_vals,mode = "L")
        label = self.FER_TO_UNIFIED[int(self.labels[idx])]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_class_weights(self):
        unified_labels = [self.FER_TO_UNIFIED[l] for l in self.labels]
        counts = Counter(unified_labels)
        total = sum(counts.values())
        weights = torch.zeros(7)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total/(7*count)
        return weights
    
def get_dataloaders(dataset_name : str = "fer2013"):
    print(f"\nLoading {dataset_name.upper()}")
    if dataset_name == "fer2013":
        train_ds = FER2013Dataset("train", transform = get_transforms("train"))
        val_ds = FER2013Dataset("val", transform = get_transforms("val"))
        test_ds = FER2013Dataset("test", transform = get_transforms("test"))
        class_weights = train_ds.get_class_weights()
    else:
        raise ValueError(f"Unknown dataset : {dataset_name}. We will add rafdb later")
    
    train_loader = DataLoader(
        train_ds,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY
    )
    print(f"Train Batches : {len(train_loader)}")
    print(f"Val Batches : {len(val_loader)}")
    print(f"Test Batches : {len(test_loader)}")

    return train_loader,val_loader,test_loader,class_weights
