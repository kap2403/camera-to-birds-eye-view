import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import torch
import torchvision.models as models
from tqdm import tqdm
import json
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import CustomImageDataset
from model import UNet

train_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((128,256)),
  #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

import argparse
 
parser = argparse.ArgumentParser(description ='Search some files')
 
parser.add_argument('front_images' , help = 'front_images_path')
parser.add_argument('bev_occulsion' , help = 'bev_occulsion_path')
args = parser.parse_args()


front_image = args.front_images
bev_occulsion = args.bev_occulsion
dataset=CustomImageDataset(front_image,bev_occulsion,transform = train_transforms)

train_set , validation_set , test_Set = torch.utils.data.random_split(dataset, [72,1000,2100])

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 10, shuffle = True, num_workers = 0)

validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size = 10, shuffle = True, num_workers = 0)

test_loader=torch.utils.data.DataLoader(
    test_Set, batch_size = 10, shuffle = True, num_workers = 0)


model=UNet()

#model = UNet()
#checkpoint = torch.load('/content/drive/MyDrive/model/my_model_epoch39.pth')
#model_state_dict = checkpoint['model_state_dict']
#optimizer_state_dict = checkpoint['optimizer_state_dict']
#epoch = checkpoint['epoch']
#train_loss = checkpoint['train_loss']


model = model.to(device=device) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001) 

#model.load_state_dict(model_state_dict)
#optimizer.load_state_dict(optimizer_state_dict)


model.train()
MAX_EPOCHS = 1
for epoch in range(MAX_EPOCHS):
    running_loss = 0.0
    torch.cuda.empty_cache()
    for input_tensor, labels in tqdm(train_loader):
        images = input_tensor.to(device).float()
        labels = labels.to(device).float()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()/len(train_loader)
    print('epoch', epoch, 'loss', running_loss)