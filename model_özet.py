# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:33:29 2024

@author: Abdurrezzak ŞIK
"""

# MODEL ÖZETİ
import torchsummary as summ
import torchinfo as info

def özet(model,içerik):
    
    return info.summary(model,içerik)
    