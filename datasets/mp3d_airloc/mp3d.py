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
                 return_points: bool = True
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

        self.rgb_data = []
        self.seg_data = []
        self.depth_data = []
        self.points_data = []
        self.room = []

        for dataset in self.datasets:
            print("Dataset Name = ", dataset)
            dataset_path = os.path.join(self.base_dir, dataset)

            for scene in os.listdir(dataset_path):
                print("Scene Name = ", scene)
                scene_path = os.path.join(dataset_path, scene)

                for room_name in os.listdir(os.path.join(scene_path, "rooms")):
                    print("Room Name = ", room_name)
                    room_path = os.path.join(scene_path, "rooms", room_name)
                    raw_data_folder = os.path.join(room_path, "raw_data/")
                    points_dir = os.path.join(room_path, "points/")
                    if not os.path.isdir(points_dir):
                        print("Points not available !!!")
                    
                    ids = ids_from_folder(raw_data_folder)
                    for id in ids:
                        self.rgb_data.append(os.path.join(raw_data_folder,(str(id)+"_rgb.png")))
                        self.seg_data.append(os.path.join(raw_data_folder,(str(id)+"_instance-seg.png")))
                        self.depth_data.append(os.path.join(raw_data_folder,(str(id)+"_depth.png")))
                        self.points_data.append(os.path.join(points_dir,(str(id)+".pkl")))
                        self.room.append((dataset+scene+room_name))
                    
        self.num_images = len(self.rgb_data)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        
        """Code to return the given Images"""      
        output = []
        if self.return_img:
            rgb_img = cv2.imread(self.rgb_data[idx])
            rgb_img = torch.from_numpy(rgb_img).type(torch.float16)
            rgb_img = rgb_img.permute(2,0,1)
            rgb_img /= 255
            output.append[rgb_img]
                
        if self.return_seg:
            img = cv2.imread(self.seg_data[idx],cv2.IMREAD_ANYDEPTH)
            mask = img_to_mask(img)
            output.append(mask)
            
        if self.return_points:
            with open(self.points_data[idx],'rb') as fp:
                points = pickle.load(fp)
            output.append(points)
            
        if self.return_depth:
            print("Not Implemented error")
            # output.append()
            
        room_name = self.room[idx]
        output.append([room_name,self.rgb_data[idx]])
        
        return tuple(output)
        

def ids_from_folder(data_path):
    #Given a folder this function extracts the ids of diferent images present in the folder
    query_list=os.listdir(data_path)
    qry_ids=[]
    for i in range(len(query_list)):
        qry_ids.append(int(query_list[i].split('_')[0]))
    qry_ids=np.array(qry_ids)
    qry_ids=np.unique(qry_ids)
    return qry_ids

def img_to_mask(img):
    obj_ids = np.unique((img))
    ann_mask = []
    for i,id in enumerate(obj_ids):
        mask = {}
        mask["mask"] = torch.from_numpy(np.where(img==id,1,0)).type(torch.float16)
        mask["id"] = id
        ann_mask.append(mask)
        
    return ann_mask

def concatenate_masks_from_dictionary(ann_mask):
    #Code to get numpy array of stacked ann_masks (For Visulaisation code in utils/viz)
    for i,mask in enumerate(ann_mask):
        if i == 0 :
            print(mask["ann_mask"].shape)
            masks= mask["ann_mask"][np.newaxis,:,:]
        else :
            masks = np.vstack((masks,mask["ann_mask"][np.newaxis,:,:]))


