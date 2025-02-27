#%%
"This code extracts features from all the slices and saves it as a .h5 file per patient"

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
import numpy as np
from tqdm import tqdm
import h5py
import os
import os
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import groupby
import sys
import argparse

sys.path.append("../")
from dataDL import GroupedImageFolder, GroupedImageFolderNLST

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


parser = argparse.ArgumentParser(description='Process dataset and output directory.')
parser.add_argument('--dataset_path', type=str, required=True, 
                    help='Path to the dataset directory.')
parser.add_argument('--output_dir', type=str, required=True, 
                    help='Path to the output directory.')

args = parser.parse_args()

dataset_path = args.dataset_path
output_dir = args.output_dir
if not os.path.exists(output_dir): os.makedirs(output_dir)

t_mean = [0.485, 0.456, 0.406]
t_sd = [0.229, 0.224, 0.225]
normal_transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),   # this makes me nervous.. why not to RGB?
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(t_mean,t_sd)])


dataset = GroupedImageFolder(dataset_path, transform=normal_transform)
imgdata_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

checkpoint = "/path/to/model/segment-anything-2/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

#model = build_sam2(model_cfg, checkpoint)
model = build_sam2_video_predictor(model_cfg, checkpoint)
model = model.image_encoder.cuda()

feature_dict = {}
for key, images in tqdm(imgdata_loader):
    key = key[0]  # Extract string from list (due to batch_size=1)
    if os.path.exists(f"{output_dir}/{key}.h5"):
        continue
    features = []
    for img in images:
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            logits = model(img)
            vlogits = logits['vision_features']
        features.append(logits['vision_features'].flatten(2).transpose(1,2).mean(1).cpu())

    # Save features for this group
    with h5py.File(f"{output_dir}/{key}.h5", 'w') as f:
        f["feats"] = torch.cat(features).numpy()
# %%
