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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize_image(img, shape, interpolation=cv2.INTER_CUBIC):
    # resize relevant image axis to length of corresponding target axis while preserving aspect ratio
    axis = 0 if float(shape[0]) / float(img.shape[0]) > float(shape[1]) / float(img.shape[1]) else 1
    factor = float(shape[axis]) / float(img.shape[axis])
    img = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=interpolation)

    # crop other image axis to match target shape
    center = img.shape[int(not axis)] / 2.0
    step = shape[int(not axis)] / 2.0
    left = int(center-step)
    right = int(center+step)
    if axis == 0:
        img = img[:, left:right]
    else:
        img = img[left:right, :]

    return img


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self,front_image_dir,bev_image_dir,transform=None, target_transform=None):
        self.front_image_dir=front_image_dir
        self.bev_image_dir=bev_image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.front_images_path=list()
        self.bev_images_path=list()
        for i in tqdm(os.listdir(self.front_image_dir)):
            self.front_images_path.append(os.path.join(self.front_image_dir,i))
            self.bev_images_path.append(os.path.join(self.bev_image_dir,i))
            
    def __len__(self):
        return len(self.bev_images_path)

    def __getitem__(self, idx):
      
        img = cv2.cvtColor(cv2.imread(self.front_images_path[idx]), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(img)
    
        bev_image = cv2.cvtColor(cv2.imread(self.bev_images_path[idx]), cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            bev_image = self.transform(bev_image)

        return image, bev_image