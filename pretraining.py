# Standard library imports
import os
import random
import re
from functools import partial
from typing import List

import math

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms.v2 as v2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, EffectiveSEModule, LayerNorm2d
from timm.models import register_model, build_model_with_cfg, named_apply, generate_default_cfgs
import timm

from rdnet import RDNet

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
print(f"Data Configuration :{data_config},\nData Transformation: {transform},\nTest Transformation: {test_transform}")


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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

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

def rdnet_tiny(pretrained=False, num_classes=1000, checkpoint_path=None, device="cpu", **kwargs):
    n_layer = 7
    model_args = {
        "num_init_features": 64,
        "growth_rates": [64, 104, 128, 128, 128, 128, 224],
        "num_blocks_list": [3] * n_layer,
        "is_downsample_block": (None, True, True, False, False, False, True),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block", "Block", "BlockESE", "BlockESE", "BlockESE", "BlockESE", "BlockESE"],
        "num_classes": num_classes,
    }

    model = RDNet(**{**model_args, **kwargs})

    if pretrained:
        assert checkpoint_path is not None, "Please provide checkpoint_path for pretrained weights"
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)

    return model


model = rdnet_tiny(pretrained=True, checkpoint_path="rdnet_tiny_pretrained.pth", device = device)
model.reset_classifier(num_classes=10)
model.to(device)


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_rdnet_tiny(model, train_loader, val_loader, test_loader, device, num_epochs=20, 
                    initial_lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), 
                    save_interval=10, resume_path=None, model_name_prefix='rdnet_tiny_transfer_learn',
                    early_stopping_patience=10, run_name="model", sub_dir_seed_number='seed_29'):
    model.to(device)
    warmup_epochs = int(math.ceil((10/100)*num_epochs))
    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=betas, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()


    best_val_loss = float('inf')
    correspondin_val_acc = 0.0
    start_epoch = 0
    early_stopping_counter = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = history['val_loss'][-1]
        correspondin_val_acc = history['val_acc'][-1]
        print(f"Resuming training from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss = running_loss / total
            train_acc = 100. * correct / total

            
            val_loss, val_acc = evaluate(model, val_loader, device)


            test_loss, test_acc = evaluate(model, test_loader, device)

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%")

     
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                correspondin_val_acc = val_acc
                early_stopping_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }
                os.makedirs(f'{run_name}/{sub_dir_seed_number}', exist_ok=True)
                save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}', f"{model_name_prefix}__valLoss{val_loss:.4f}_valAcc{correspondin_val_acc:.2f}.pth")
                torch.save(checkpoint, save_path)
                print(f"Saved best checkpoint at {save_path}")
            else:
                early_stopping_counter += 1
                print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")

            scheduler.step()
            
            # Save periodic checkpoints
            if epoch % save_interval == 0:
                os.makedirs(f'{run_name}/{sub_dir_seed_number}/periodic', exist_ok=True)
                save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/periodic', f"{model_name_prefix}_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }, save_path)
                print(f"Saved periodic checkpoint at {save_path}")

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                os.makedirs(f'{run_name}/{sub_dir_seed_number}/earlystopping', exist_ok=True)
                save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/earlystopping', f"{model_name_prefix}_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }, save_path)
                print(f"Saved periodic checkpoint at {save_path}")
                break

        os.makedirs(f'{run_name}/{sub_dir_seed_number}/last_epoch', exist_ok=True)
        save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/last_epoch', f"{model_name_prefix}_epoch{epoch}.pth")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history
        }, save_path)
        return history, correspondin_val_acc

    except KeyboardInterrupt:
        print("Training interrupted! Saving checkpoint...")
        os.makedirs(f'{run_name}/{sub_dir_seed_number}/KeyboardInterrupt', exist_ok=True)
        save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/KeyboardInterrupt', f"{model_name_prefix}_interrupted_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }, save_path)
        print(f"Checkpoint saved at {save_path}. You can resume training from here.")
        return history, correspondin_val_acc
    

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

# Make sure num_classes is correct here
model = rdnet_tiny(pretrained=False, num_classes=10)
checkpoint = torch.load("./model/seed_29/rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth", map_location=device, weights_only=False)

checkpoint.keys()

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# device = "cpu"
model.eval()
model.to(device)
inputs, targets = next(iter(val_loader))
inputs, targets = inputs.to(device), targets.to(device)
outputs = model(inputs)
_, preds = outputs.max(1)

print("Predictions:", preds[:10])
print("Targets:    ", targets[:10])
# device = "cuda"

def evaluate_model(model, test_loader, device, class_names=None):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f"F1 Score (Macro):    {f1_macro:.4f}")
    print(f"F1 Score (Micro):    {f1_micro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels = class_names if class_names else "auto", yticklabels = class_names if class_names else "auto")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
    
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

print(50*"="+"\tStats for val loader\t"+"="*50)
evaluate_model(model, val_loader, device, class_names)
print(50*"="+"\tStats for test loader\t"+"="*50)
evaluate_model(model, test_loader, device, class_names)
print(50*"="+"\tStats for train loader\t"+"="*50)
evaluate_model(model, train_loader, device, class_names)
