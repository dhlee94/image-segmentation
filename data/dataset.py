import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from timm.models.layers import to_2tuple

class ImageDataset(Dataset):
    def __init__(self, Image_path, num_class=1, one_hot=True, transform=None):
        self.Image_path = Image_path
        self.transform = transform
        self.num_class = num_class
        self.one_hot = one_hot

    def __len__(self):
        return len(self.Image_path['image'])

    def _one_hot_encoding(self, data):
        H, W = data.shape
        new_data = np.zeros((self.num_class, H, W))
        for idx in range(self.num_class):
            new_data[np.where(data==idx)] = 1
        return new_data

    def _not_one_hot_encoding(self, data):
        N, H, W = data.shape
        new_data = np.zeros((H, W))
        for idx in range(N):
            new_data[np.where(data[idx, ...]==1)] = idx+1
        return new_data

    def __getitem__(self, idx):
        data = Image.open(self.Image_path['image'][idx]).convert('RGB')
        target = Image.open(self.Image_path['label'][idx]).convert('L')
        #target = self.one_hot_encoding(target)
        if self.transform != None:
            transformed = self.transform(image=np.array(data), mask=np.array(target)//255)
            return transformed['image'], transformed['mask'].unsqueeze(0)
        else:
            return torch.FloatTensor(data.permute(2, 0, 1)), torch.LongTensor(target).unsqueeze(0)