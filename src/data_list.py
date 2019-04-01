# from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path as osp

from torchvision.datasets.folder import IMG_EXTENSIONS

from utils import *

def make_dset_list(filename='../data/train.txt', path='../data/train'):
    file = open(filename, 'w')
    path_list = os.listdir(path)
    for label in path_list:
        p = osp.join(path, label)
        for i, j, k in os.walk(p):
            for img in k:
                tag = osp.join(i, img) + ' ' + label
                file.write(tag + '\n')
    file.close()

def make_dataset(image_list, labels, datadir):
    if labels:
        LEN = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(LEN)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(osp.join(datadir, val.split()[0]), np.array([int(la) for la in val.split()[1:]])) for val in
                      image_list]
        else:
            images = [(osp.join(datadir + val.split()[0]), int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, datadir=None, transform=None, target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels, datadir)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

    def __len__(self):
        return len(self.imgs)


class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

    def __len__(self):
        return len(self.imgs)
