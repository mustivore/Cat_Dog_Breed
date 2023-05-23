import glob
import os
import torch
from PIL import Image
from torchvision.transforms import *
import numpy as np


class CatDogDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, split, label_dict):
        assert split in ["test_set", "training_set"]
        self.image_dir = "./dataset"

        path = os.path.join(self.image_dir, split + "/*/*.jpg")
        self.images = glob.glob(path)

        if split == "training_set":
            self.transform = Compose([transforms.Resize((224, 224)),
                                      transforms.ColorJitter(hue=.05, saturation=.05),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                      ])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        self.labels = list(label_dict.keys())

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""

        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        np.array(img)
        img = self.transform(img)

        label = img_path.split("\\")[2]
        label = self.labels.index(label)
        label = np.array(label)
        return img, label
