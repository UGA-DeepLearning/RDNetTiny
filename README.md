# Transfer Learning, Quantization, and Robustness Testing on RDNetTiny

This repository contains a set of Jupyter notebooks that demonstrate key stages of deep learning model optimization and evaluation using the RDNetTiny architecture from Hugging Face, pretrained on ImageNet-1k, and fine-tuned on the CIFAR-10 dataset.

## 📁 Notebooks Overview

### 1. `pretrainingrdnet.ipynb`
This notebook performs **transfer learning** by fine-tuning the pretrained RDNetTiny model from Hugging Face on the CIFAR-10 dataset. It includes:
- Dataset preparation and augmentation
- Model loading and customization for CIFAR-10
- Training with learning rate scheduling and performance monitoring
- Evaluation of the fine-tuned model

### 2. `RDNetQuantization.ipynb`
This notebook applies **post-training quantization** techniques to the fine-tuned RDNetTiny model to reduce model size and improve inference speed. It includes:
- Dynamic and static quantization methods using Onnxruntime
- Accuracy evaluation and size/speed comparison with the original model

### 3. `RDNetGaussianTesting.ipynb`
This notebook evaluates the **robustness** of the fine-tuned RDNetTiny model under common adversarial attacks. It includes:
- FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) attacks
- Visualization of perturbed examples
- Accuracy degradation analysis under adversarial conditions

---

## 📦 Requirements

All dependencies are installed directly within each notebook using `pip`. No separate setup or environment configuration is needed.



