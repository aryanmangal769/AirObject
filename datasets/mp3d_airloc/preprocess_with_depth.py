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
import json


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

base_dir ="/data/datasets/aryan/x-view"
datasets = ["mp3d"]
depth = True

data = {}
data["images"] = []
for dataset in datasets:
    print("Dataset Name = ", dataset)
    dataset_path = os.path.join(base_dir, dataset)

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
                #Getting Image paths
                rgb_data = os.path.join(raw_data_folder,(str(id)+"_rgb.png"))
                seg_data = os.path.join(raw_data_folder,(str(id)+"_instance-seg.png"))
                depth_data = os.path.join(raw_data_folder,(str(id)+"_depth.png"))
                points_data = os.path.join(points_dir,(str(id)+".pkl"))
                room = (dataset+scene+room_name)

                #Reading Images. Semantic and points for this case
                image = []                        
                img = cv2.imread(seg_data,cv2.IMREAD_ANYDEPTH)
                mask = img_to_mask(img)
                image.append(mask)
                
                with open(points_data,'rb') as fp:
                    points = pickle.load(fp)
                image.append(points)

                depth = cv2.imread(depth_data,cv2.IMREAD_ANYDEPTH)
                image.append(depth)
                    
                image.append([room,rgb_data])

                #Generating Object Points from it 
                ann_masks, points, depth, roomname = image[0], image[1], image[2] ,image[3]
                
                keypoints = points['points']
                descriptors = points['point_descs']

                
                image_objects = {}
                image_objects['points'] = []
                image_objects['descs'] = []
                image_objects['depths'] = []
                image_objects['ids'] = [] 
                image_objects['room_image_name'] = roomname

                for a in range(len(ann_masks)):
                    ann_mask = ann_masks[a]['mask']
                    object_filter = ann_mask[keypoints[:,0].T,keypoints[:,1].T]
                    np_obj_pts = keypoints[np.where(object_filter==1)[0]].numpy()

                    obj_id = str(ann_masks[a]['id']) 
                    depth_value = np.mean(depth[np.where(ann_mask==1)])

                    image_objects['depths'].append(depth_value)
                    image_objects['points'].append(keypoints[np.where(object_filter==1)[0]].float())
                    image_objects['descs'].append(descriptors[np.where(object_filter==1)[0]].float())
                    image_objects['ids'].append(obj_id) 
                
                data["images"].append(image_objects)


save_json_file = "/data/datasets/aryan/x-view/images.pkl"
with open(save_json_file, "wb") as out_file:
    pickle.dump(data, out_file)

