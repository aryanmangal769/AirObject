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

from model.build_model import build_gcn, build_netvlad, build_seqnet, build_airobj ,build_airloc
from model.graph_models.gcn import GCN

from datasets.mp3d_airloc.mp3d_triplet_v2 import mp3d
from datasets.utils.batch_collator import eval_custom_collate

from statistics import mean

import time

def batch_gnn_processing(batch_obj,model,device):
    batch_features = []
    #This is a temporary function, need to replace this with GCN descriptor similar to NetVlad Descriptor for batched processing.
    for obj in batch_obj:
        nodes = torch.tensor(obj.shape[0])
        adj = torch.ones(nodes,nodes)
        output= model(obj.to(device), adj.to(device))
        batch_features.append(output.squeeze())
    return torch.stack(batch_features)


def points_to_obj_desc(batch_objects,netvlad_model,device):
    batch_decs = []
    for image_objects in batch_objects:
        #Takes only those objects whone no. points are more then rejection threshold
        batch_points = []
        for object_points in image_objects['descs']:
            if object_points.shape[0]>3:
                batch_points.append(object_points.to(device))
        #Just a temporary code we need to remove this kinf of images either form dataset or from pkl file
<<<<<<< HEAD
        if len(batch_points) <= 1:
=======
        if len(batch_points) == 0:
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b
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


def train(configs):
    #files config
    base_dir = configs['base_dir']
    datasets = configs['datasets']
    log_dir = configs['log_dir']
<<<<<<< HEAD
    model_save_dir = configs["airloc_save_path"]
=======
    
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b

    ## Train config
    train_config = configs['model']['airloc']
    seqlen = train_config['train']['seqlen']
    batch_size = train_config['train']['batch_size']
    epochs = train_config['train']['epochs']
    lr = train_config['train']['lr']
    checkpoint = train_config['train']['checkpoint']
    lambda_d = train_config['train']['lambda_d'] 
    
    ##Model Config
    dropout = train_config['dropout']
    nfeat = train_config['descriptor_dim']
    nhid = train_config['hidden_dim']
    nclass = train_config['nout']
    
<<<<<<< HEAD
    configs['num_gpu'] = [1]
=======
    configs['num_gpu'] = [2]
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b
    configs['public_model'] = 0
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    shuffle_dataset = True
    validation_split = .2
    random_seed= 42

    dataset = mp3d(base_dir=base_dir,datasets=datasets )
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

<<<<<<< HEAD
    # train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,num_workers = 8,sampler=train_sampler)
    # test_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,num_workers = 8,sampler=valid_sampler)
=======
    # train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,num_workers = 16,sampler=train_sampler)
    # test_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,num_workers = 12,sampler=valid_sampler)
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b
        
    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,sampler=train_sampler)
    test_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=eval_custom_collate,sampler=valid_sampler)
        
    netvlad_model = build_netvlad(configs)
    netvlad_model.eval()
    
    model = build_airloc(configs)
    model.train()

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
<<<<<<< HEAD
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
=======
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_airobj'))
    
    logdir = writer.file_writer.get_logdir()
    save_dir = os.path.join(logdir, 'saved_model')
    os.makedirs(save_dir, exist_ok=True)
    
    sum_iter = 0
    for epoch in tqdm(range(epochs), desc='train'):
        train_accuracy = []
        for step, (anchor_pts, positive_pts, negative_pts) in enumerate(tqdm(train_loader)):
            
            anchor_objs = points_to_obj_desc(anchor_pts,netvlad_model,device)
            positive_objs = points_to_obj_desc(positive_pts,netvlad_model,device)
            negative_objs = points_to_obj_desc(negative_pts,netvlad_model,device)

            if anchor_objs==0 or positive_objs==0 or negative_objs==0:
                continue

            anchor_room = model(anchor_objs)
            positive_room = model(positive_objs)
            negative_room = model(negative_objs)

            loss = triplet_loss(anchor_room, positive_room, negative_room)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(anchor_room,positive_room,negative_room)
            train_accuracy.append(acc.item())
            
            sum_iter+=1
            
            writer.add_scalar('Train/Loss', loss, sum_iter)
        
        print("Train_accuracy : ", mean(train_accuracy) )
            
        test_accuracy = []
        for step, (anchor_pts, positive_pts, negative_pts) in enumerate(tqdm(test_loader)):
            
            anchor_objs = points_to_obj_desc(anchor_pts,netvlad_model,device)
            positive_objs = points_to_obj_desc(positive_pts,netvlad_model,device)
            negative_objs = points_to_obj_desc(negative_pts,netvlad_model,device)

<<<<<<< HEAD
            if anchor_objs==0 or positive_objs==0 or negative_objs==0:
                continue

=======
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b
            anchor_room = model(anchor_objs)
            positive_room = model(positive_objs)
            negative_room = model(negative_objs)
            
            loss = triplet_loss(anchor_room, positive_room, negative_room)
            
            acc = accuracy(anchor_room,positive_room,negative_room)
            test_accuracy.append(acc.item())
            
            sum_iter+=1
            
            writer.add_scalar('Train/Loss', loss, sum_iter)
        
        print("Test_accuracy : ", mean(test_accuracy) )
<<<<<<< HEAD

        torch.save(model.state_dict(), model_save_dir)
=======
    
>>>>>>> 0dcc9a4c213a4057ad7f9b504d4baabc1499378b
    writer.close()

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

    train(configs)
    
if __name__ == "__main__":
    main()