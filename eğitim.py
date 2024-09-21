# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:40:11 2024

@author: Abdurrezzak ŞIK
"""

# Kütüphaneler 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import time
import torch.nn.functional as F
import torchsummary as summ
#import sys
#sys.path.append('model_özet.py')  # model_özet.py'nin bulunduğu dizin

from model_özet import özet

from model_test import test

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VERİ ÖNİŞLEME
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# VERİ YÜKLEME

train_dataset = ImageFolder(root="train",transform=transform)
val_dataset = ImageFolder(root="val",transform=transform)
test_dataset = ImageFolder(root="test",transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)

# MODEL OLUŞTURMA

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        # CONV KATMANLARI
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 4 ,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 4 ,stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4 ,stride=2,padding=1)
        
        # FULLY CONNECTED KATMANLARI
        self.fc1 = nn.Linear(in_features = 128*4*4, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 10) # len(train_loader.dataset)
            
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,kernel_size=2,stride=2,padding=1)
        
        # Flattening
        x = x.view(-1,128*4*4)
        
        # Ann Katmanları
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
model = CNN()
#özet(model,(3,224,244))
    
            
# HİPERPARAMETRELER

optimizer = optim.Adam(model.parametres(),lr=2e-4)
criterion = nn.CrossEntropyLoss()


# MODEL EĞİTME 

print("Eğitim Başlıyor...")

Epochs = 10
train_accuracies = []
val_accuracies = []
train_losses = []

for epoch in range(Epochs):
    
    start_time = time.process_time_ns()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    all_train_preds  = []
    all_train_labels = []
    
    for i,data in enumerate(train_loader,1):
        train_inputs,train_labels = data
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_loss = criterion(train_outputs,train_labels)
        train_loss.backward()
        optimizer.step()
        
        _,train_preticted = torch.max(train_outputs.data,1)
        train_correct +=(train_outputs == train_preticted).sum().item()
        train_total +=train_labels.size(0)
        train_running_loss += train_loss.item()
        
        all_train_preds.append(train_preticted.numpy())
        all_train_labels.append(train_labels.numpy())
        
    train_accuracy = 100*(train_correct/train_total)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_running_loss / len(train_loader.dataset))
    
    # MODEL DEĞERLENDİRME
    
    with torch.no_grad():
        
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        all_val_labels = []
        all_val_preds  = []
        
        for data in val_loader:
            val_inputs,val_labels = data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs,val_labels)
            val_loss.backward()
            
            _,val_preticted = torch.max(val_outputs.data,1)
            val_correct = (val_preticted == val_outputs).sum().item()
            val_total = val_labels.size()
            val_running_loss += val_loss.item()
            
            all_val_preds.append(val_preticted.numpy())
            all_val_labels.append(val_labels.numpy())
            
    val_accuracy = 100*(val_correct/val_total)
    val_accuracies.append(val_accuracy)
    val_running_loss.append(val_running_loss / len(val_dataset.dataset))
    end_time = time.process_time_ns()
    
    print(f" Epoch : {epoch+1} / {Epochs}| Train_Accuraccy : {train_accuracy:.2f} | Val_Accuracy : {val_accuracy:.2f} | Train_Loss : {train_loss:.2f} | Val_Loss : {val_loss:.2f} | Time : {(end_time-start_time):.f} ")
            

test(model, test_loader, device, criterion, test_dataset)



























