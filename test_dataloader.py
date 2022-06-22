import os
import pickle
import yaml
import argparse

import torch
from torch.utils import data
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm

from model.build_model import build_gcn, build_netvlad, build_seqnet, build_airobj

from datasets.mp3d_airloc.mp3d import mp3d
from datasets.utils.batch_collator import eval_custom_collate

dataset = mp3d(base_dir="/home/aryan/Mp3d_dataset/x_view",datasets= ["mp3d"])

loader = data.DataLoader(dataset=dataset, batch_size=1, collate_fn=eval_custom_collate,shuffle=False)
batch = next(iter(loader))



