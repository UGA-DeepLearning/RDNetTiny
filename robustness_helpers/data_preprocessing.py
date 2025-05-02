import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode
from .config import DATA_ROOT


# timm_model = timm.create_model('rdnet_tiny.nv_in1k', pretrained=True)
# data_config = timm.data.resolve_model_data_config(timm_model)
# transform = timm.data.create_transform(**data_config, is_training=True)
# test_transform = timm.data.create_transform(**data_config, is_training=False)
# torch.save(timm_model.state_dict(), 'rdnet_tiny_pretrained.pth')

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

def get_cifar10_loaders(batch_size=64, val_size=5000, data_root=DATA_ROOT, seed=29):
    full_dataset = CIFAR10(root=data_root, train=True, download=True)
    targets = np.array(full_dataset.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_cifar10_full_dataset_and_indices(val_size=5000, data_root=DATA_ROOT, seed=29):
    full_dataset = CIFAR10(root=data_root, train=True, download=True)
    targets = np.array(full_dataset.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    return full_dataset, train_idx, val_idx