import os
import pickle
import yaml
import argparse
from datetime import datetime

import sys
sys.path.append('.')

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm
from scipy.spatial.distance import cdist


from model.build_model import build_gcn, build_netvlad, build_seqnet, build_airobj ,build_airloc
from model.graph_models.gcn import GCN

from datasets.mp3d_airloc.mp3d_triplet_v2 import mp3d
from datasets.utils.batch_collator import eval_custom_collate

from statistics import mean

import time



def points_to_obj_desc(batch_objects,netvlad_model,device):
    batch_decs = []
    for image_objects in batch_objects:
        #Takes only those objects whone no. points are more then rejection threshold
        batch_points = []
        for object_points in image_objects['descs']:
            if object_points.shape[0]>3:
                batch_points.append(object_points.to(device))
        #Just a temporary code we need to remove this kinf of images either form dataset or from pkl file
        if len(batch_points) <= 1:
            print(image_objects['room_image_name'])
            return 0
        object_desc = netvlad_model(batch_points)
        batch_decs.append(object_desc)
    return batch_decs

def accuracy(a,p,n):
    dist1 = (a - p).pow(2).sum(1)
    dist2 = (a - n).pow(2).sum(1)
    pred = (dist1 - dist2 ).cpu()
    return (pred > 0).sum()*1.0/dist1.size()[0]


def eval(configs):
    #files config
    base_dir = configs['base_dir']
    datasets = configs['datasets']

    batch_size = configs['batch_size']
    
    configs['num_gpu'] = [1]
    configs['public_model'] = 0

    ref_path = configs["ref_path"]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    shuffle_dataset = True
    validation_split = .2
    random_seed= 42

    dataset = mp3d(base_dir=base_dir,datasets=datasets,train = False )
    test_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,shuffle =True)
        
    netvlad_model = build_netvlad(configs)
    netvlad_model.eval()
    
    model = build_airloc(configs)
    model.eval()
    
    with open(ref_path,'rb') as f:
        ref = pickle.load(f)
    
    rooms = []
    ref_data = []
    for key in ref.keys():
        rooms.append(key)
        ref_data.append(ref[key].cpu().detach().numpy())

    test_accuracy = []
    for step, anchor_pts in enumerate(tqdm(test_loader)):
        anchor_objs = points_to_obj_desc(anchor_pts,netvlad_model,device)
        if anchor_objs == 0:
            continue
        anchor_room = model(anchor_objs)
        query = anchor_room.cpu().detach().numpy()

        dMat = cdist(ref_data,query,"euclidean")
        mInds = np.argsort(dMat,axis=0)[:1]
        mInds = mInds.reshape(query.shape[0])

        positive = 0
        for i, obj in enumerate(anchor_pts):
            query_room = obj["room_image_name"][0]
            # print("query_room",query_room)
            ref_room = rooms[mInds[i]]
            # print(ref_room)
            if (ref_room==query_room):
                positive+=1
        
        acc= positive/len(anchor_objs)
        test_accuracy.append(acc)
    
    print("Test_accuracy : ", mean(test_accuracy) )


def main():
    parser = argparse.ArgumentParser(description="Training AirLoc")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 

        default = ""
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int, 
        default = 1
    )
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['use_gpu'] = args.gpu

    eval(configs)
    
if __name__ == "__main__":
    main()