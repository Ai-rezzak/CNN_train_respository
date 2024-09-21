import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model_özet import özet



#  RESNET MODELİ

def ResNet(num_classes):
    # Modeli yükle
    model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    
    # FC katmanının giriş boyutunu alın
    num_fltrs = model.fc.in_features
    
    # Yeni FC katmanı
    model.fc = nn.Linear(num_fltrs, num_classes)
    
    # Tüm katmanların ağırlıklarını dondur
    for param in model.parameters():
        param.requires_grad = False
        
    # Sadece son katmanın ağırlıklarını eğitilebilir yap
    for param in model.fc.parameters():
        param.requires_grad = True
        
    # Optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    return optimizer, model

######################################################################################################

# AlexNET MODELİ

def AlexNet(num_classes):
    
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    num_fltrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_fltrs, num_classes) 
    
    # Tüm katmanların ağırlıklarını dondur
    for param in model.parameters():
        param.requires_grad = False
        
    # Sadece son katmanın ağırlıklarını eğitilebilir yap
    for param in model.fc.parameters():
        param.requires_grad = True
        
    # Optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    return optimizer, model

######################################################################################################
 
# VGGNET MODELİ

def VGGNet(num_classes):
    
    model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
    num_fltrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_fltrs, num_classes) 
    
    # Tüm katmanların ağırlıklarını dondur
    for param in model.parameters():
        param.requires_grad = False
        
    # Sadece son katmanın ağırlıklarını eğitilebilir yap
    for param in model.fc.parameters():
        param.requires_grad = True
        
    # Optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    return optimizer, model

######################################################################################################

# DENSENET MODELİ
def DenseNET(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_fltrs =model.classifier.in_features
    model.classifier = nn.Linear(num_fltrs,num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adadelta(model.classifier.parameters(), lr=1e-2)

    return optimizer , model

    

    
    