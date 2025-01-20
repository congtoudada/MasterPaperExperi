import os
import glob
import random
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class VideoTransform:
    def __init__(self, size, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BICUBIC):
        if not isinstance(scale, tuple) or not len(scale) == 2:
            raise ValueError('Scale should be a tuple with two elements.')

        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img1, img2):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img1, self.scale, self.ratio)
        img1 = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        img2 = F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)

        if random.random() < 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)

        img1 = self.normalize(transforms.ToTensor()(img1))
        img2 = self.normalize(transforms.ToTensor()(img2))

        return img1, img2