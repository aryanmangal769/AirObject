#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from typing import Optional, Union
import cv2
import os
import random
import numbers
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import sys
from tqdm import tqdm
sys.path.append('.')
import pickle
import random




class mp3d(Dataset):
    def __init__(self,
                 base_dir: str,
                 datasets: list,
                 seqlen: int = 4,
                 dilation: Optional[int] = None,
                 stride: Optional[int] = None,
                 height: int = 480,
                 width: int = 640,
                 *,
                 return_img: bool = False,
                 return_seg: bool = True,
                 return_depth: bool = False,
                 return_points: bool = True,
                 train: bool = True
                 ):
        self.base_dir = base_dir
        self.datasets = datasets
        self.seqlen = seqlen
        self.dilation = dilation
        self.height = height
        self.width = width
        self.return_img = return_img
        self.return_seg = return_seg
        self.return_depth = return_depth
        self.return_points = return_points
        self.is_train = train

        self.rgb_data = []
        self.seg_data = []
        self.depth_data = []
        self.points_data = []
        self.room = []

        with open(self.base_dir,'rb') as f:
            images = pickle.load(f)
        self.images = images["images"]
        self.num_images = len(self.images)

        self.room = [self.images[i]["room_image_name"][0] for i in range(self.num_images)]
        self.points = [self.images[i]["points"] for i in range(self.num_images)]
        self.descs = [self.images[i]["descs"] for i in range(self.num_images)]
        self.ids = [self.images[i]["ids"] for i in range(self.num_images)]
                    

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        """Code to return the data in Image format Images"""  
        
        anchor = self.images[idx]
        
        if self.is_train:
            positive_idxs = np.where(np.array(self.room) == self.room[idx])[0]
            negative_idxs = np.where(np.array(self.room) != self.room[idx])[0]
            
            positive_idx = random.choice(positive_idxs)
            positive = self.images[positive_idx]

            negative_idx = random.choice(negative_idxs)
            negative = self.images[negative_idx]
            
            return anchor,positive,negative
        
        else:
<<<<<<< HEAD
            return [anchor]
=======
            return anchor
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b

