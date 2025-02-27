"This code extracts features from all the slices and saves it as a .h5 file per patient"

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
import numpy as np
from tqdm import tqdm
import h5py
import os 
import sys
sys.path.append("../")
from dataDL import GroupedImageFolder, GroupedImageFolderNLST, GroupedImageFolderRadFM
import glob
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

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
dataset = GroupedImageFolder(dataset_path, transform=preprocess)

imgdata_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
model = model.cuda()
if not os.path.exists(output_dir): os.mkdir(output_dir)
for key, images in tqdm(imgdata_loader):
    key = key[0] # Extract string from list (due to batch_size=1)
    if os.path.exists(f"{output_dir}/{key}.h5"):
        continue
    features = []
    for img in images:
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            logits = model.encode_image(img)
        try:
            features.append(logits.cpu())
        except Exception as exc:
            print('error:', exc)
            print(key)

    
    # Save features for this group
    with h5py.File(f"{output_dir}/{key}.h5", 'w') as f:
        f["feats"] = torch.cat(features).numpy()