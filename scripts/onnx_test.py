import onnxruntime as ort

# Load the quantized model
session = ort.InferenceSession("quantized_rdnet_tiny_model.onnx", providers=["CPUExecutionProvider"])

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define your dataset (e.g., CIFAR-10)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

import numpy as np
import torch
import time

correct = 0
total = 0

start_time = time.time()

for images, labels in test_loader:
    # Convert to numpy
    images = images.numpy()
    
    # ONNX expects input in shape (batch_size, channels, height, width)
    ort_inputs = {session.get_inputs()[0].name: images}
    ort_outs = session.run(None, ort_inputs)
    
    outputs = torch.tensor(ort_outs[0])
    _, predicted = torch.max(outputs, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

end_time = time.time()

accuracy = 100 * correct / total
inference_time = (end_time - start_time) / total  # seconds per image

print(f'Accuracy of quantized model on CIFAR-10 test images: {accuracy:.2f}%')
print(f'Average Inference Time per Image: {inference_time*1000:.2f} ms')
print(f'Total Inference Time: {(end_time - start_time):.2f} seconds')

import os

model_path = "quantized_rdnet_tiny_model.onnx"
model_size = os.path.getsize(model_path) / (1024 * 1024)  # size in MB
print(f"Model size: {model_size:.2f} MB")
