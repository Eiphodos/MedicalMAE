# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
import torch.distributed as dist

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import multiprocessing as mp
from itertools import repeat
from datetime import datetime
from PIL import Image
from torchvision import datasets, transforms

import utils
from mae_transforms import ClipCTIntensity


class DataAugmentationForMAE(object):
    def __init__(self, args):
        if args.data_set == 'DeepLesion':
            mean = (0)
            std = (1)
        elif args.data_set == 'ImageNet':
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = 0.5
            std = 0.5
        if args.clip_ct_intensity:
            self.transform = transforms.Compose([
                ClipCTIntensity(args.ct_intensity_min, args.ct_intensity_max),
                transforms.RandomResizedCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])

    def __call__(self, image):
        transformed_image = self.transform(image)
        return transformed_image

    def __repr__(self):
        repr = "(DataAugmentationForMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def build_mae_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'DeepLesion':
        root = os.path.join(args.data_path, 'train')
        img_folder = os.getenv('TMPDIR')
        if utils.is_main_process():
            extract_dataset_to_local(root, img_folder)
        if args.distributed:
            dist.barrier()
        return datasets.folder.ImageFolder(img_folder, loader=deeplesion_loader, transform=transform)
    elif args.data_set == 'ImageNet':
        root = args.data_path
        img_folder = os.getenv('TMPDIR')
        if utils.is_main_process():
            extract_dataset_to_local(root, img_folder)
        if args.distributed:
            dist.barrier()
        return datasets.folder.ImageFolder(img_folder, transform=transform)
    else:
        return datasets.folder.ImageFolder(args.data_path, transform=transform)


def deeplesion_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        nimg = np.array(img)
        nimg = nimg.astype(np.uint8)
        img = Image.fromarray(nimg)
        return img.convert('L')


def extract_dataset_to_local(root, image_folder):
    root, dirs, files = next(os.walk(root))
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    pool.starmap(extract_npz_to_disk, zip(files, repeat(root), repeat(image_folder)))
    pool.close()
    pool.join()


def extract_npz_to_disk(file, root, image_folder):
    case_folder = os.path.join(image_folder, file[:-4])
    os.makedirs(case_folder, exist_ok=True)
    data = np.load(os.path.join(root, file))
    for i, arr in enumerate(data):
        filename = file[:-4] + '_' + str(i) + '.png'
        im = Image.fromarray(data[arr])
        im.save(os.path.join(case_folder, filename))
