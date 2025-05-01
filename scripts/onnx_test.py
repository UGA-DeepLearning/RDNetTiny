import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import time
import os
import csv
import timm 
from rdnet import rdnet_tiny

# Define dataset and transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Function to evaluate model
def evaluate_model(model_func, model_name, is_onnx=False):
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in test_loader:
        if is_onnx:
            images_np = images.numpy()
            ort_inputs = {model_func.get_inputs()[0].name: images_np}
            ort_outs = model_func.run(None, ort_inputs)
            outputs = torch.tensor(ort_outs[0])
        else:
            images = images.to('cpu')
            outputs = model_func(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = (end_time - start_time) / total
    total_time = end_time - start_time
    return accuracy, inference_time, total_time

# -----------------------
# Evaluate quantized model
# -----------------------
onnx_model_path = "quantized_rdnet_tiny_model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
quant_accuracy, quant_inf_time, quant_total_time = evaluate_model(session, "quantized_rdnet", is_onnx=True)
quant_size = os.path.getsize(onnx_model_path) / (1024 * 1024)

# -----------------------
# Evaluate original model
# -----------------------
pth_model_path = "rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth"
model = timm.create_model("rdnet_tiny", pretrained=False, num_classes=10)
model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
model.eval()
orig_accuracy, orig_inf_time, orig_total_time = evaluate_model(model, "rdnet_tiny", is_onnx=False)
orig_size = os.path.getsize(pth_model_path) / (1024 * 1024)

# -----------------------
# Save results
# -----------------------
results = [
    {
        "Model": "Quantized ONNX",
        "Accuracy (%)": f"{quant_accuracy:.2f}",
        "Inference Time per Image (ms)": f"{quant_inf_time * 1000:.2f}",
        "Total Inference Time (s)": f"{quant_total_time:.2f}",
        "Model Size (MB)": f"{quant_size:.2f}"
    },
    {
        "Model": "Original PyTorch",
        "Accuracy (%)": f"{orig_accuracy:.2f}",
        "Inference Time per Image (ms)": f"{orig_inf_time * 1000:.2f}",
        "Total Inference Time (s)": f"{orig_total_time:.2f}",
        "Model Size (MB)": f"{orig_size:.2f}"
    }
]

csv_file = "results.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Evaluation complete. Results saved to {csv_file}.")
