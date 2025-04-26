import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts

from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms.v2 as v2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from evaluate import evaluate
from train import train_rdnet_tiny
from rdnet import rdnet_tiny, RDNet

import re


from functools import partial
from typing import List

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models import register_model, build_model_with_cfg, named_apply, generate_default_cfgs
from timm.layers import DropPath
from timm.layers import LayerNorm2d
import timm

if torch.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Device: {device}")

seed = 29
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

DATA_ROOT = './data.cifar10'
CIFAR10_MEAN = [0.49139968, 0.48215841,  0.44653091]
CIFAR10_STD = [0.24703223,  0.24348513,  0.26158784]

timm_model = timm.create_model('rdnet_tiny.nv_in1k', pretrained=True)
data_config = timm.data.resolve_model_data_config(timm_model)
transform = timm.data.create_transform(**data_config, is_training=True)
test_transform = timm.data.create_transform(**data_config, is_training=False)
# print(f"Data Configuration :{data_config},\nData Transformation: {transform},\nTest Transformation: {test_transform}")


torch.save(timm_model.state_dict(), 'rdnet_tiny_pretrained.pth')

from torchvision import transforms
from torchvision.transforms import InterpolationMode

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(224, 224),
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=(0.6, 1.4),
        contrast=(0.6, 1.4),
        saturation=(0.6, 1.4)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

test_transform = transforms.Compose([
    transforms.Resize(
        size=248,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# trainset = CIFAR10(root='./data.cifar10', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# testset = CIFAR10(root='./data.cifar10', train=False, download=True, transform=test_transform)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader

full_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True)
targets = np.array(full_dataset.targets)

sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=29)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform   = test_transform

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)#, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)#, num_workers=2)

test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)#, num_workers=2)

__all__ = ["RDNet"]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


rdnet_tiny_cfg = {
    "url": "",  # optional: local path to weights if needed
    "num_classes": 1000,
    "input_size": (3, 224, 224),
    "crop_pct": 0.9,
    "interpolation": "bicubic",
    "mean": IMAGENET_DEFAULT_MEAN,
    "std": IMAGENET_DEFAULT_STD,
    "first_conv": "stem.0",
    "classifier": "head.fc",
}



model = rdnet_tiny(pretrained=True, checkpoint_path="rdnet_tiny_pretrained.pth", device = device)
model.reset_classifier(num_classes=10)
model.to(device)

if __name__ == "__main__":
    train_rdnet_tiny(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=20,
    initial_lr=1e-3,
    weight_decay=0.05,
    betas=(0.9, 0.999),
    save_interval=10,
    resume_path=None,
    model_name_prefix='rdnet_tiny_transfer_learn',
    early_stopping_patience=10,
    run_name="model",
    sub_dir_seed_number='seed_29'
)


# # # Make sure num_classes is correct here
# # model = rdnet_tiny(pretrained=False, num_classes=10)
# # checkpoint = torch.load("./model/seed_29/rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth", map_location=device, weights_only=False)

# # checkpoint.keys()

# # model.load_state_dict(checkpoint["model_state_dict"])
# # model.to(device)

# # device = "cpu"
# # model.eval()
# # model.to(device)
# # inputs, targets = next(iter(val_loader))
# # inputs, targets = inputs.to(device), targets.to(device)
# # outputs = model(inputs)
# # _, preds = outputs.max(1)

# # print("Predictions:", preds[:10])
# # print("Targets:    ", targets[:10])
# # device = "cuda"

# from evaluate import evaluate_model
    
# class_names = [
#     'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
# ]

# print(50*"="+"\tStats for val loader\t"+"="*50)
# evaluate_model(model, val_loader, device, class_names)
# print(50*"="+"\tStats for test loader\t"+"="*50)
# evaluate_model(model, test_loader, device, class_names)
# print(50*"="+"\tStats for train loader\t"+"="*50)
# evaluate_model(model, train_loader, device, class_names)
