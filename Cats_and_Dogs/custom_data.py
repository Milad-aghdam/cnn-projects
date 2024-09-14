import matplotlib.pyplot as plt
import torch
import torchvision 
from torch.utils.data import Dataset
from PIL import Image
import os 
import numpy as np
class CustomCatsDogsDataset():
    def __init__(self, root_dir, transform=None):
        """
            Args:
                root_dir (string): Directory with all the images, separated into 'cat' and 'dog' folders.
                transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """Helper function to load image paths and assign labels"""
        cat_dir = os.path.join(self.root_dir, 'cats')
        dog_dir = os.path.join(self.root_dir, 'dogs')

        for file_name in os.listdir(cat_dir):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):  # Make sure we are loading image files
                self.image_paths.append(os.path.join(cat_dir, file_name))
                self.labels.append(0)

        for file_name in os.listdir(dog_dir):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):  # Make sure we are loading image files
                self.image_paths.append(os.path.join(dog_dir, file_name))
                self.labels.append(1) 
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label
