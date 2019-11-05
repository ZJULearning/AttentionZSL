import os
from os.path import join
import numpy as np
import torch

from torch.utils.data import Dataset
from util import load_pilimage
from torchvision import transforms

def build_transform(methods, im_size):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform_default = transforms.Compose([transforms.ToTensor(), norm])
    if methods == "resize_crop":
        transform = transforms.Compose([
            transforms.Resize(int(im_size[0] * 8 / 7.)),
            transforms.CenterCrop(im_size[0]),
            transforms.ToTensor(),
            norm])
    elif methods == "FiveCrop":
        transform = transforms.Compose([
            transforms.Resize(int(im_size[0] * 8 / 7.)),
            transforms.FiveCrop(im_size[0]),
            transforms.Lambda(lambda  crops: torch.stack([norm(transforms.ToTensor()(crop)) for crop in crops]))
            ])
    elif methods == "TenCrop":
        transform = transforms.Compose([
            transforms.Resize(int(im_size[0] * 8 / 7.)),
            transforms.TenCrop(im_size[0]),
            transforms.Lambda(lambda  crops: torch.stack([norm(transforms.ToTensor()(crop)) for crop in crops]))
            ])
    elif methods == "TenCrop+Resize":
        transform = [transforms.Compose([
            transforms.Resize(int(im_size[0] * 8 / 7.)),
            transforms.TenCrop(im_size[0]),
            transforms.Lambda(lambda  crops: torch.stack([norm(transforms.ToTensor()(crop)) for crop in crops]))
            ]), transform_default]
    return transform

class EvaluationFactory(Dataset):
    def __init__(
        self,
        data_path,
        methods, 
        im_size
    ):

        self.methods = methods
        self.im_size = im_size
        self.transform = build_transform(methods, im_size)
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        ims = load_pilimage(self.data_path[idx], None, crop_im=False)
        if self.methods == "TenCrop+Resize":
            resize_ims = load_pilimage(self.data_path[idx], self.im_size, crop_im=False)
            return torch.cat(
                [self.transform[0](ims), self.transform[1](resize_ims)[None]]
            )
        else:
            ims = load_pilimage(self.data_path[idx], None, crop_im=False)
            return self.transform(ims)
