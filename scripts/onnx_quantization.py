import torch
import onnx
import numpy as np
import onnxruntime as ort
from rdnet import rdnet_tiny
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

import os

# Step 1: Load your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "rdnet_tiny_transfer_learn__valLoss0.1992_valAcc93.86.pth"

# Initialize and load the model
model = rdnet_tiny(pretrained=False, num_classes=10)  # Replace with your model
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)
model.eval()  # Set the model to evaluation mode

print("Model loaded and set to evaluation mode.")

# Step 2: Export the model to ONNX format
onnx_path = "rdnet_tiny_model.onnx"
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust input shape if needed
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], verbose=True)

print(f"Model exported to {onnx_path}")

# Define a custom calibration data reader if needed
class MyCalibrationDataReader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.batch_index = 0  # Initialize the batch index

    def get_next(self):
        if self.batch_index < len(self.data_loader):
            batch = next(iter(self.data_loader))  # Get the next batch from the DataLoader
            self.batch_index += 1
            # Ensure the batch is in the format expected by ONNX Runtime
            inputs = {'input': batch[0].numpy()}  # batch[0] should be the image tensor
            return inputs
        else:
            return None  # Return None if there are no more batches



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
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader to batch your data
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Now use this dataset in your calibration data reader
calibration_data_reader = MyCalibrationDataReader(data_loader)



# Step 3: Quantize the model using ONNX Runtime
def quantize_model(input_model_path, output_model_path):
    # Apply post-training quantization using ONNX Runtime
    # Update the quantization function call, removing `quant_type`
    quantized_model = quantize_static(
        input_model_path,
        output_model_path,
        calibration_data_reader=calibration_data_reader
    )

    print(f"Quantized model saved to {output_model_path}")

# Step 4: Quantize and save the quantized model
input_model_path = onnx_path
output_model_path = "quantized_rdnet_tiny_model.onnx"
quantize_model(input_model_path, output_model_path)

# Step 5: Run inference with the quantized model using ONNX Runtime
def run_inference(onnx_model_path):
    # Load the quantized ONNX model using ONNX Runtime
    sess = ort.InferenceSession(onnx_model_path)

    # Create dummy input data (use the same shape as the original model's input)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Perform inference
    inputs = {sess.get_inputs()[0].name: dummy_input}
    outputs = sess.run(None, inputs)

    # Print the output (inference result)
    print("Inference result:", outputs)

# Step 6: Perform inference with the quantized model
run_inference(output_model_path)
