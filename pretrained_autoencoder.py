# import the necessary packages
import json
import os
import random

import numpy as np
import scipy.io
import torch
from PIL import Image
from torchvision import transforms

from torch import nn
from torchscope import scope
from torchvision import models

num_classes = 196

data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CarRecognitionModel(nn.Module):
    def __init__(self):
        super(CarRecognitionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # self.pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        #x = self.pool(x)
        x = x.view(-1, 2048)  # [N, 2048]
        #x=1
        x = self.fc(x)
        #x = self.softmax(x)
        return x

def init_autoencoder(): 
    filename = 'car_recognition.pt'
    model = CarRecognitionModel()
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model.eval()
    return model

def pretrained_autoencoder_similarity(autoencoder, current_cropped, prev_feature, prev_cropped, data_transforms=data_transforms): 
    transformer = data_transforms['valid']
    with torch.no_grad():
        if not prev_feature:
            prev_transf = transformer(prev_cropped).unsqueeze(dim=0)
            prev_feature = autoencoder(prev_transf)
        current_transf = transformer(current_cropped).unsqueeze(dim=0)    
        current_feature = autoencoder(current_transf)
    cossim = torch.nn.CosineSimilarity()
    sim = cossim(current_feature, prev_feature)
    return sim, current_feature