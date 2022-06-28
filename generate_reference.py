import os
import pickle
import yaml
import argparse

import torch
from torch.utils import data
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm
import shutil

import sys
sys.path.append('.')

from datasets.preprocess_with_depth import preprocess
from model.build_model import build_gcn, build_netvlad, build_seqnet, build_airobj ,build_airloc

def points_to_obj_desc(batch_objects,netvlad_model,device):
    batch_decs = []
    for image_objects in batch_objects:
        #Takes only those objects whone no. points are more then rejection threshold
        batch_points = []
        for object_points in image_objects['descs']:
            if object_points.shape[0]>3:
                batch_points.append(object_points.to(device))
        #Just a temporary code we need to remove this kinf of images either form dataset or from pkl file
        if len(batch_points) == 0:
            print(image_objects['room_image_name'])
            return 0
        object_desc = netvlad_model(batch_points)
        batch_decs.append(object_desc)
    return batch_decs

def generate_ref_filesystem(configs):
    base_dir = configs["base_dir"]
    datasets = configs["datasets"]
    ids = configs["ids"]

    for dataset in datasets:
        print("Dataset Name = ", dataset)
        dataset_path = os.path.join(base_dir, dataset)
        ref_dataset_path = os.path.join(base_dir,dataset+"_ref")
        os.makedirs(ref_dataset_path,exist_ok=True)

        for scene in os.listdir(dataset_path):
            print("Scene Name = ", scene)
            scene_path = os.path.join(dataset_path, scene)
            ref_scene_path = os.path.join(ref_dataset_path,scene)
            os.makedirs(ref_scene_path,exist_ok=True)
            ref_rooms_path = os.path.join(ref_scene_path,"rooms")
            os.makedirs(ref_rooms_path,exist_ok=True)

            for room_name in os.listdir(os.path.join(scene_path, "rooms")):
                print("Room Name = ", room_name)
                room_path = os.path.join(scene_path, "rooms", room_name)
                ref_room_path = os.path.join(ref_scene_path, "rooms", room_name)
                os.makedirs(ref_room_path,exist_ok=True)
                raw_data_folder = os.path.join(room_path, "raw_data/")
                ref_raw_data_folder = os.path.join(ref_room_path, "raw_data/")
                os.makedirs(ref_raw_data_folder,exist_ok=True)
                points_dir = os.path.join(room_path, "points/")
                ref_points_dir = os.path.join(ref_room_path, "points/")
                os.makedirs(ref_points_dir,exist_ok=True)
            
                for id in ids:
                        rgb_data = os.path.join(raw_data_folder,(str(id)+"_rgb.png"))
                        seg_data = os.path.join(raw_data_folder,(str(id)+"_instance-seg.png"))
                        depth_data = os.path.join(raw_data_folder,(str(id)+"_depth.png"))
                        points_data = os.path.join(points_dir,(str(id)+".pkl"))

                        ref_rgb_data = os.path.join(ref_raw_data_folder,(str(id)+"_rgb.png"))
                        ref_seg_data = os.path.join(ref_raw_data_folder,(str(id)+"_instance-seg.png"))
                        ref_depth_data = os.path.join(ref_raw_data_folder,(str(id)+"_depth.png"))
                        ref_points_data = os.path.join(ref_points_dir,(str(id)+".pkl"))

                        shutil.copyfile(rgb_data, ref_rgb_data)
                        shutil.copyfile(seg_data, ref_seg_data)
                        shutil.copyfile(depth_data, ref_depth_data)
                        shutil.copyfile(points_data, ref_points_data)


def generate(configs):

    generate_ref_filesystem(configs)

    datasets = configs["datasets"]
    configs["datasets"] = [dataset+"_ref" for dataset in datasets]
    configs['num_gpu'] = [2]
    configs['public_model'] = 0

    netvlad_model = build_netvlad(configs)
    netvlad_model.eval()

    model = build_airloc(configs)
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if os.path.isfile(configs["pkl_path"]):
        with open(configs["pkl_path"],'rb') as f:
            images = pickle.load(f)
    else:
        images = preprocess(configs)
    sorted_images = {}
    for image in images["images"]:
        if image["room_image_name"][0] not in sorted_images.keys():
            sorted_images[image["room_image_name"][0]] = []
        sorted_images[image["room_image_name"][0]].append(image)

    room_descriptors = {}
    for key in sorted_images.keys():
        objs = points_to_obj_desc(sorted_images[key],netvlad_model,device)
        room = model(objs)
        key = key.replace("_ref","")
        room_descriptors[key] = torch.mean(room,0)
    
    pkl_path = configs["ref_pkl_path"]
    with open(pkl_path, "wb") as out_file:
        pickle.dump(room_descriptors, out_file)



def main():
    parser = argparse.ArgumentParser(description="Training AirLoc")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 
        default = ""
    )
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)

    generate(configs)
    
if __name__ == "__main__":
    main()