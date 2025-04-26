import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model

# Step 1: Recreate the model architecture
model = create_model("rdnet_tiny", pretrained=False, num_classes=10)

# Step 2: Load checkpoint
checkpoint = torch.load('/content/RDNetTiny/scripts/rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth', weights_only=True)

# Step 3: Extract only the model weights
state_dict = checkpoint['model_state_dict']

# Step 4: Load weights into the model
model.load_state_dict(state_dict, strict=False)

# Define transforms: resize CIFAR-10 images to 224x224 to match ResNet's input
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load CIFAR-10 test set
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cpu')  # keep on CPU for quantization

# Testing accuracy before quantization
model.eval()
model.to(device)

# Measure accuracy before quantization
correct = 0
total = 0

start_time = time.time()  # Start timing inference
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
end_time = time.time()  # End timing inference

time_before_quantization = end_time - start_time
accuracy_before_quantization = 100 * correct / total
print(f"Model accuracy before quantization: {accuracy_before_quantization:.2f}%")
print(f"Inference time before quantization: {time_before_quantization:.4f} seconds")

# Save model size before quantization
torch.save(model.state_dict(), "model_before_quantization.pth")
model_size_before_quantization = os.path.getsize("model_before_quantization.pth") / (1024 ** 2)  # Convert to MB
print(f"Model size before quantization: {model_size_before_quantization:.2f} MB")


# --- Quantization ---
# Apply quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

# Testing accuracy after quantization
model.eval()
model.to(device)

# Measure accuracy after quantization
correct = 0
total = 0

start_time = time.time()  # Start timing inference
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
end_time = time.time()  # End timing inference

time_after_quantization = end_time - start_time
accuracy_after_quantization = 100 * correct / total
print(f"Model accuracy after quantization: {accuracy_after_quantization:.2f}%")
print(f"Inference time after quantization: {time_after_quantization:.4f} seconds")

# Save model size after quantization
torch.save(model.state_dict(), "model_after_quantization.pth")
model_size_after_quantization = os.path.getsize("model_after_quantization.pth") / (1024 ** 2)  # Convert to MB
print(f"Model size after quantization: {model_size_after_quantization:.2f} MB")
