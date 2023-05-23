import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms


class DogBreedDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, split, label_dict):
        assert split in ["train", "val"]

        self.image_dir = "imagewoof2-320"

        path = os.path.join(self.image_dir, split + "/*/*.JPEG")
        self.images = glob.glob(path)

        # resize the image for our network (bc he accepts only 224*224 images)
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

        # for example: imagewoof2-320/train/n02111889/ILSVRC2012_val_00032819.JPEG the label is n02111889
        label = img_path.split("\\")[2]
        label = self.labels.index(label)
        label = np.array(label)
        return img, label
