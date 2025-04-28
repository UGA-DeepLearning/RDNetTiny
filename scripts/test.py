import torch
import time
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from rdnet import rdnet_tiny, quantizable_rdnet_tiny, QuantizableBlock, QuantizableBlockESE 
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
  

def get_calibration_loader(train_dataset, batch_size=32, num_samples=512):
    indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

def evaluate_model(model, test_loader, device='cpu', max_samples=None):
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if max_samples is not None and i * test_loader.batch_size >= max_samples:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size_bytes = os.path.getsize("temp.pth")
    os.remove("temp.pth")
    return size_bytes / (1024 * 1024)  # Convert to MB

def measure_inference_time(model, input_tensor, num_runs=100, warmup=10):
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    return (time.time() - start_time) / num_runs * 1000  # ms per inference

def prepare_for_static_quantization(model, calibration_loader):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Fuse modules
    for module in model.modules():
        if isinstance(module, (QuantizableBlock, QuantizableBlockESE)):
            torch.quantization.fuse_modules(module.layers, 
                [['0', '1', '2'], ['3', '4', '5'], ['6', '7']], inplace=True)
    
    prepared_model = torch.quantization.prepare(model)
    
    # Calibrate
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            _ = prepared_model(inputs.to('cpu'))
    
    return prepared_model

def main():
    # Setup data loaders
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )

    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    calibration_loader = get_calibration_loader(train_dataset)

    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = "rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth"

    # Original model
    original_model = rdnet_tiny(pretrained=False, num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    original_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    original_model.to(device)
    original_model.eval()

    # Quantizable model
    quantizable_model = quantizable_rdnet_tiny(pretrained=False, num_classes=10)
    quantizable_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    quantizable_model.to(device)
    quantizable_model.eval()

    # Create test input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Benchmark original model
    print("Benchmarking original model...")
    original_acc = evaluate_model(original_model, test_loader, device, max_samples=1000)
    original_size = get_model_size(original_model)
    original_time = measure_inference_time(original_model, dummy_input)

    # Dynamic quantization
    print("\nApplying dynamic quantization...")
    dynamic_model = torch.quantization.quantize_dynamic(
        original_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    dynamic_acc = evaluate_model(dynamic_model, test_loader, device, max_samples=1000)
    dynamic_size = get_model_size(dynamic_model)
    dynamic_time = measure_inference_time(dynamic_model, dummy_input)

    # Static quantization
    print("\nApplying static quantization...")
    static_model = quantizable_model.to('cpu')
    prepared_model = prepare_for_static_quantization(static_model, calibration_loader)
    static_model = torch.quantization.convert(prepared_model)
    static_acc = evaluate_model(static_model, test_loader, 'cpu', max_samples=1000)
    static_size = get_model_size(static_model)
    static_time = measure_inference_time(static_model, dummy_input.to('cpu'))

    # Print results
    print("\nResults:")
    print(f"{'Metric':<20} | {'Original':>10} | {'Dynamic Q':>10} | {'Static Q':>10}")
    print("-" * 50)
    print(f"{'Accuracy (%)':<20} | {original_acc:>10.2f} | {dynamic_acc:>10.2f} | {static_acc:>10.2f}")
    print(f"{'Size (MB)':<20} | {original_size:>10.2f} | {dynamic_size:>10.2f} | {static_size:>10.2f}")
    print(f"{'Latency (ms)':<20} | {original_time:>10.2f} | {dynamic_time:>10.2f} | {static_time:>10.2f}")

if __name__ == "__main__":
    main()