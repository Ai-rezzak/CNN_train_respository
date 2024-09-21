import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import time
from model_özet import özet
from model_test import test
from my_models import *


# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VERİ ÖNİŞLEME
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# VERİ YÜKLEME
dataset = ImageFolder(root="data", transform=transform)
data_size = len(dataset)
train_size = int(0.8 * data_size)
val_size = int(0.1 * data_size)
test_size = data_size - train_size - val_size

trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
val_loader = DataLoader(valset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=True)



# HİPERPARAMETRELER
criterion = nn.CrossEntropyLoss()
num_classes = len(dataset.classes)

#optimizer, model = ResNet(num_classes)
#optimizer , model = AlexNet(num_classes)
#optimizer , model = VGGNet(num_classes)
optimizer , model = DenseNET(num_classes)


# MODEL EĞİTME
print("Eğitim Başlıyor...")

Epochs = 10
train_accuracies = []
val_accuracies = []
train_losses = []

özet(model=model,içerik=(1,3,224,224))

for epoch in range(Epochs):
    start_time = time.process_time_ns()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for i, data in enumerate(train_loader, 1):
        train_inputs, train_labels = data
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_loss = criterion(train_outputs, train_labels)
        train_loss.backward()
        optimizer.step()
        
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_correct += (train_predicted == train_labels).sum().item()
        train_total += train_labels.size(0)
        train_running_loss += train_loss.item()
        
    train_accuracy = 100 * (train_correct / train_total)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_running_loss / len(train_loader.dataset))
    
    # MODEL DEĞERLENDİRME
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        for data in val_loader:
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)
            val_running_loss += val_loss.item()
    
    val_accuracy = 100 * (val_correct / val_total)
    val_accuracies.append(val_accuracy)
    val_running_loss /= len(val_loader)

    end_time = time.process_time_ns()
    print(f"Epoch : {epoch + 1} / {Epochs} | Train_Accuracy : {train_accuracy:.2f} | Val_Accuracy : {val_accuracy:.2f} | Train_Loss : {train_losses[-1]:.2f} | Val_Loss : {val_running_loss:.2f} | Time : {(end_time - start_time) * 1e-9:.2f} ")

test(model, test_loader, device, criterion, test_loader)
