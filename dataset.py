# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 07:30:21 2022

Modifications by orukundo@gmail.com Olivier Rukundo
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class FFGPROJECTDataset(Dataset):
    def __init__(self, trainingImages, trainingMasks, transform=None):
        self.trainingImages = trainingImages
        self.trainingMasks = trainingMasks
        self.transform = transform
        self.images = os.listdir(trainingImages)
        self.maskss = os.listdir(trainingMasks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.trainingImages, self.images[index])
        mask_path = os.path.join(self.trainingMasks, self.maskss[index])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

