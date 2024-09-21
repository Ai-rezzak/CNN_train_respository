# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:34:49 2024

@author: Abdurrezzak ŞIK
"""

# MODEL TEST ETME 
import torch

test_accuracies = []

def test(model, test_loader, device, criterion, test_dataset):
    
    test_correct = 0
    test_total = 0
    test_running_loss = 0.0
    
    all_test_preds  = []
    all_test_labels = []
    
    model.eval()
    
    with torch.no_grad():
        for data in test_loader:
            test_inputs, test_labels = data
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            
            _, test_predicted = torch.max(test_outputs.data, 1)
            
            test_correct += (test_predicted == test_labels).sum().item()
            test_total += test_labels.size(0)
            test_running_loss += test_loss.item()
            
            all_test_preds.append(test_predicted.numpy())
            all_test_labels.append(test_labels.numpy())
    
    test_accuracy = 100 * (test_correct / test_total)
    test_accuracies.append(test_accuracy)
    
    test_loss_value = test_running_loss / len(test_loader)
    
    print(f"Test_Accuraccy : %{test_accuracy:.2f} | Test_Loss : {test_loss:.2f} ")
    
    return test_loss_value, test_accuracy

       