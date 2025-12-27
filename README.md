<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=CNN%20Training%20Framework&fontSize=60&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=Complete%20Deep%20Learning%20Training%20Pipeline&descAlignY=55" width="100%"/>
</div>

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
  
</div>

<h3 align="center">🧠 Comprehensive CNN Training and Evaluation System</h3>

<p align="center">
  A professional deep learning framework featuring custom CNN architectures, transfer learning, and model evaluation tools for image classification tasks.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [File Structure](#-file-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architectures](#-model-architectures)
- [Training Parameters](#-training-parameters)
- [Results and Metrics](#-results-and-metrics)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This repository provides a complete training framework designed to solve image classification problems using **Convolutional Neural Networks (CNN)**. It supports both training models from scratch and fine-tuning pre-trained models with transfer learning.

### 🌟 Core Objectives

- ✅ Build flexible and modular CNN architectures
- ✅ Custom model design and training
- ✅ Rapid model development with transfer learning
- ✅ Comprehensive model evaluation and testing
- ✅ Visualization and performance analysis

---

## ✨ Features

<table>
  <tr>
    <td width="50%">
      
### 🎨 Custom Model Design
- Flexible layer configuration
- Customizable activation functions
- Dropout and batch normalization support
- Modular architecture design

    </td>
    <td width="50%">
      
### 🚀 Transfer Learning
- Pre-trained model integration
- Fine-tuning mechanisms
- Feature extraction
- Multi-model comparison

    </td>
  </tr>
  <tr>
    <td width="50%">
      
### 📊 Comprehensive Training
- Automatic checkpoint system
- Early stopping mechanism
- Learning rate scheduling
- Data augmentation support

    </td>
    <td width="50%">
      
### 🔍 Model Evaluation
- Detailed performance metrics
- Confusion matrix visualization
- Model comparison tools
- Test batch analysis

    </td>
  </tr>
</table>

---

## 📁 File Structure

```
CNN_train_respository/
│
├── 📄 my_models.py            # Module containing custom CNN architectures
├── 📄 eğitim.py               # Main model training script
├── 📄 transfer_eğitim.py      # Transfer learning training pipeline
├── 📄 model_test.py           # Model testing and evaluation tools
├── 📄 model_özet.py           # Model summary and statistics
└── 📄 README.md               # Project documentation
```

### 📄 File Descriptions

| File | Description | Key Functions |
|------|-------------|---------------|
| **my_models.py** | Contains custom CNN architectures | Model classes, layer definitions |
| **eğitim.py** | Trains models from scratch | Training loop, data loading, checkpointing |
| **transfer_eğitim.py** | Fine-tunes pre-trained models | Transfer learning, feature extraction |
| **model_test.py** | Tests trained models | Accuracy, precision, recall calculation |
| **model_özet.py** | Displays model details | Parameter count, layer information |

---

## 🚀 Installation

### Requirements

```bash
Python 3.8+
PyTorch 1.10+
torchvision
numpy
matplotlib
scikit-learn
pillow
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ai-rezzak/CNN_train_respository.git
cd CNN_train_respository
```

### Step 2: Install Required Libraries

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

### Step 3: Prepare Your Dataset

Organize your dataset in the following format:

```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
└── test/
    ├── class1/
    ├── class2/
    └── class3/
```

---

## 💻 Usage

### 1️⃣ Custom Model Training

```python
# Run the eğitim.py file
python eğitim.py

# To customize parameters:
python eğitim.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

**Basic Parameters:**
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--data_path`: Dataset path
- `--save_path`: Model save path

### 2️⃣ Training with Transfer Learning

```python
# Run the transfer_eğitim.py file
python transfer_eğitim.py

# With pre-trained model selection:
python transfer_eğitim.py --model resnet50 --freeze_layers 7
```

**Transfer Learning Parameters:**
- `--model`: Pre-trained model (resnet18, resnet50, vgg16, vgg19)
- `--freeze_layers`: Number of layers to freeze
- `--fine_tune`: Fine-tuning mode (True/False)

### 3️⃣ Model Testing and Evaluation

```python
# Test model performance with model_test.py
python model_test.py --model_path saved_models/best_model.pth

# For detailed analysis:
python model_test.py --model_path saved_models/best_model.pth --visualize True
```

### 4️⃣ View Model Summary

```python
# Examine model details with model_özet.py
python model_özet.py --model_path saved_models/best_model.pth
```

---

## 🏗️ Model Architectures

### Custom CNN Model

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
```

### Supported Pre-trained Models

| Model | Parameters | Features |
|-------|------------|----------|
| **ResNet-18** | ~11M | Fast training, good performance |
| **ResNet-50** | ~25M | Deeper, higher accuracy |
| **VGG-16** | ~138M | Classic architecture, strong feature extraction |
| **VGG-19** | ~143M | Deepest VGG model |

---

## ⚙️ Training Parameters

### Hyperparameter Configuration

```python
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'scheduler': 'StepLR',
    'step_size': 10,
    'gamma': 0.1
}
```

### Data Augmentation

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## 📊 Results and Metrics

### Evaluation Metrics

- ✅ **Accuracy**: Overall accuracy rate
- ✅ **Precision**: Precision score
- ✅ **Recall**: Recall score
- ✅ **F1-Score**: Harmonic mean
- ✅ **Confusion Matrix**: Class-based performance
- ✅ **Loss Curves**: Training and validation loss graphs

### Sample Output

```
Epoch [50/50]
Train Loss: 0.0234 | Train Acc: 98.5%
Val Loss: 0.0456 | Val Acc: 96.2%

Test Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:  96.75%
Precision: 96.82%
Recall:    96.68%
F1-Score:  96.75%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 Examples

### Simple Training Example

```python
import torch
from my_models import CustomCNN
from eğitim import train_model

# Create model
model = CustomCNN(num_classes=10)

# Start training
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    learning_rate=0.001
)
```

### Transfer Learning Example

```python
from torchvision import models
from transfer_eğitim import fine_tune_model

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Fine-tuning
fine_tune_model(
    model=model,
    train_loader=train_loader,
    num_classes=10,
    freeze_layers=7
)
```

---

## 🎯 Use Case Scenarios

| Scenario | Recommended Approach | File |
|----------|---------------------|------|
| **Small dataset (<1000 images)** | Transfer Learning | `transfer_eğitim.py` |
| **Medium dataset (1000-10000)** | Transfer Learning + Fine-tuning | `transfer_eğitim.py` |
| **Large dataset (>10000)** | Custom Model | `eğitim.py` |
| **Rapid prototyping** | Transfer Learning | `transfer_eğitim.py` |
| **Custom architecture need** | Custom Model | `my_models.py` + `eğitim.py` |

---

## 🔧 Advanced Features

### 1. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in train_loader:
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
```

### 2. Model Ensemble

```python
models = [model1, model2, model3]
predictions = ensemble_predict(models, test_loader)
```

### 3. Gradient Visualization

```python
from model_özet import visualize_gradients
visualize_gradients(model, sample_input)
```

---

## 📈 Performance Tips

### 🚀 Training Acceleration

1. **Batch Size Optimization**: Use maximum batch size for your GPU memory
2. **Mixed Precision**: 40% speedup with 16-bit computation
3. **DataLoader Workers**: Parallel data loading with `num_workers=4`
4. **Pin Memory**: Faster GPU transfer with `pin_memory=True`

### 🎯 Accuracy Improvement

1. **Data Augmentation**: More data variety
2. **Learning Rate Scheduling**: Adaptive learning rate
3. **Ensemble Methods**: Combination of multiple models
4. **Regularization**: Use of dropout and weight decay

---

## 🐛 Troubleshooting

### Common Errors and Solutions

**1. Out of Memory (OOM) Error**
```python
# Solution: Reduce batch size
batch_size = 16  # Instead of 32
```

**2. Overfitting**
```python
# Solution: Add dropout and data augmentation
dropout_rate = 0.5
```

**3. Slow Training**
```python
# Solution: Check GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. 🔀 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔃 Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings
- Write unit tests
- Update README

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Abdurrezzak ŞIK**

[![Email](https://img.shields.io/badge/Email-rezzak.eng%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:rezzak.eng@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdurrezzak-%C5%9F%C4%B1k-64b919233/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ai-rezzak)

---

## 🌟 Acknowledgments

If you find this project useful, don't forget to give it a ⭐!

<div align="center">
  
### 📚 Related Projects

[![Dental X-Ray Detection](https://img.shields.io/badge/Dental_X--Ray_Detection-00FFFF?style=for-the-badge&logo=github&logoColor=black)](https://github.com/Ai-rezzak/dental-xray-yolov8-detection)
[![Dog Emotion Detection](https://img.shields.io/badge/Dog_Emotion_Detection-8A2BE2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ai-rezzak/dog-emotion-detection-yolov8)
[![Autonomous Vision System](https://img.shields.io/badge/Autonomous_Vision-FF6F00?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ai-rezzak/autonomous-system-vision-deep-learning)

</div>

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=120&section=footer" width="100%"/>
  
  <br>
  
  <sub>Made with ❤️ by Abdurrezzak ŞIK</sub>
  
  <br><br>
  
  <sub>"Building intelligent systems, one layer at a time" 🧠</sub>
  
</div>
