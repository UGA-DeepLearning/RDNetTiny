import torch
from .rdnet import rdnet_tiny  # or however you're importing

def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() or torch.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_cifar10_model(weights_path: str, device: str = 'cpu') -> torch.nn.Module:
    model = rdnet_tiny(pretrained=False, num_classes=10)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = get_device()
    return model.to(device)
