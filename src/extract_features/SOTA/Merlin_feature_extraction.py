#%%
"This code extracts features from all the slices and saves it as a .h5 file per patient"

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
import numpy as np
from tqdm import tqdm
import h5py
import os 
import warnings
import torch
import merlin
from glob import glob 
import argparse

parser = argparse.ArgumentParser(description='Process dataset and output directory.')
parser.add_argument('--dataset_path', type=str, required=True, 
                    help='Path to the dataset directory.')
parser.add_argument('--output_dir', type=str, required=True, 
                    help='Path to the output directory.')

args = parser.parse_args()

dataset_path = args.dataset_path
output_dir = args.output_dir


if not os.path.exists(output_dir): os.makedirs(output_dir)

model = merlin.models.Merlin()
model.eval()

datalist = [{'image': file} for file in glob(f'{dataset_path}/*.npy')]


cache_dir = dataset_path.replace("train", "train_cache")
imgdata_loader = merlin.data.DataLoader(
    datalist=datalist,
    cache_dir=cache_dir,
    batchsize=1,
    shuffle=True,
    num_workers=0,
)

#%%
model = model.cuda()
if not os.path.exists(output_dir): os.mkdir(output_dir)
torch.autograd.set_detect_anomaly(True)
for i , (filename, img) in enumerate(tqdm(imgdata_loader)): 
    filename = filename[0].split('/')[-1].replace('.npy', '')  
    if os.path.exists(f"{output_dir}/{filename}.h5"): 
      continue
    logits = model.model.encode_image(img['image'].cuda())
    logits = logits[0]
    try:
      with h5py.File(f"{output_dir}/{filename}.h5", 'w') as f:
          #f["feats"] = logits.squeeze(0).detach().cpu().numpy()
        f["feats"] = logits.detach().cpu().numpy()
    except:
      print(filename)