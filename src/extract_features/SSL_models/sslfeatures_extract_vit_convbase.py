#%%
import torch
from vits import vit_conv_base

#from vision_transformer_ibot import vit_small
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import h5py
import argparse

from dataDL import GroupedImageFolder, GroupedImageFolderNLST

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process dataset and output directory.')
parser.add_argument('--dataset_path', type=str, required=True, 
                    help='Path to the dataset directory.')
parser.add_argument('--output_dir', type=str, required=True, 
                    help='Path to the output directory.')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the model file.')

args = parser.parse_args()

# Assign arguments to variables
dataset_path = args.dataset_path
output_dir = args.output_dir
model_path = args.model_path

if not os.path.exists(output_dir): os.makedirs(output_dir)
chkpt = torch.load(model_path)
# print(chkpt.keys())
model = vit_conv_base()
print(model)
model.head=torch.nn.Identity()

msg = model.load_state_dict(chkpt,strict=False) # dangerous! But here ok, since I should have removed the head_weights from the state_dict
print(msg)
#assert len(msg.missing_keys)==0, f"There are keys missing in the state dict: {msg.missing_keys}!"
t_mean = [0.485, 0.456, 0.406]
t_sd = [0.229, 0.224, 0.225]
normal_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # this makes me nervous.. why not to RGB?
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(t_mean,t_sd)])


dataset = GroupedImageFolder(dataset_path, transform=normal_transform)
imgdata_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
model = model.cuda()
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
        features.append(logits.cpu())
        

    # Save features for this group
    with h5py.File(f"{output_dir}/{key}.h5", 'w') as f:
        f["feats"] = torch.cat(features).numpy()
# %%
