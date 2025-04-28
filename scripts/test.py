import torch
import time
from sklearn.metrics import accuracy_score
from rdnet import rdnet_tiny
from torch import nn
from torchvision import datasets, transforms
from rdnet_quantized import quantizable_rdnet_tiny
from rdnet import rdnet_tiny
import torch.quantization
import os
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.quantization import fuse_modules, prepare, convert

def fuse_model(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            modules_to_fuse = []
            for idx in range(len(module) - 1):
                if isinstance(module[idx], nn.Conv2d) and isinstance(module[idx+1], nn.BatchNorm2d):
                    modules_to_fuse.append([str(idx), str(idx+1)])
            if modules_to_fuse:
                fuse_modules(module, modules_to_fuse, inplace=True)
        else:
            fuse_model(module)  # recursive

def evaluate_model(model, test_loader, device='cpu'):
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
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

def measure_inference_time(model, input_tensor, num_runs=100):
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    return (time.time() - start_time) / num_runs * 1000  # ms per inferenc

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
path_to_cifar10_weights = "rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth"
model = quantizable_rdnet_tiny(pretrained=False, num_classes=10)
checkpoint = torch.load(path_to_cifar10_weights, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)
# 1. Define your model (assuming RDNet is already defined)
# model = rdnet_tiny(num_classes=10, checkpoint_path='rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth')  # Your original model
model.eval()  # Must be in eval mode for quantization


fuse_model(model)

# 2. Apply dynamic quantization to the whole model
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Original model
    qconfig_spec={
        torch.nn.Linear,  # Quantize Linear layers
        torch.nn.Conv2d,  # Quantize Conv2d layers
    },
    dtype=torch.qint8,  # Quantize to 8-bit integers
)

# # 3. Test inference (input remains float32)
# input_data = torch.rand(1, 3, 224, 224)  # Example float32 input
# output = quantized_model(input_data)  # Automatically handles quantization

# 1. Create sample input data and test loader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Using CIFAR-10 for testing (replace with your dataset)
test_dataset = datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

# Using CIFAR-10 for testing (replace with your dataset)
train_dataset = datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

import torch.optim as optim

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(quantized_model.parameters(), lr=1e-3)

# Save function
def save_model(model, filename="quantized_rdnet.pth"):
    torch.save(model.state_dict(), filename)
    print(f"✅ Model saved to {filename}")

# --- Training Loop ---
num_epochs = 5
save_path = "./quantized_rdnet.pth"

try:
    for epoch in range(num_epochs):
        quantized_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = quantized_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss/len(train_loader):.4f}")

except KeyboardInterrupt:
    print("\n⛔ Training interrupted. Saving model...")
    save_model(quantized_model, save_path)

# Final Save
save_model(quantized_model, save_path)
print('🏁 Training Finished')

# # 3. Create test input tensor
# dummy_input = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224

# # 4. Evaluate original model
# original_acc = evaluate_model(model, test_loader)
# original_size = get_model_size(model)
# original_time = measure_inference_time(model, dummy_input)

# # 5. Evaluate quantized model
# quantized_acc = evaluate_model(quantized_model, test_loader)
# quantized_size = get_model_size(quantized_model)
# quantized_time = measure_inference_time(quantized_model, dummy_input)

# # 6. Final comparison printout
# print("\n" + "="*60)
# print(f"{'Metric':<20} | {'Original':>12} | {'Quantized':>12} | {'Change':>12}")
# print("="*60)
# print(f"{'Accuracy (%)':<20} | {original_acc:>12.2f} | {quantized_acc:>12.2f} | {quantized_acc-original_acc:>+12.2f}")
# print(f"{'Size (MB)':<20} | {original_size:>12.2f} | {quantized_size:>12.2f} | {(quantized_size-original_size):>+12.2f}")
# print(f"{'Inference (ms)':<20} | {original_time:>12.2f} | {quantized_time:>12.2f} | {(original_time/quantized_time):>12.2f}x")
# print("="*60 + "\n")

# # 7. Additional debug info
# print("Sample input shape:", dummy_input.shape)
# print("Test dataset size:", len(test_dataset))
# print("Quantized model layer types:", 
#       set(type(m) for m in quantized_model.modules()))