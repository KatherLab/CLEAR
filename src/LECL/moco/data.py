from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

class LesionDataset(Dataset):
    def __init__(self,path,df_path,transform1,transform2,img_size=224,crop_min=0.08,orig_img_size=512):
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_files = np.random.permutation(glob(f"{path}/**/*.png",recursive=True)).tolist()
        chunks = pd.read_csv(df_path, chunksize=10**6)
        self.df = pd.concat(chunks)
        self.img_size = img_size
        self.resize = transforms.Resize([self.img_size,self.img_size])
        self.rrc = transforms.RandomResizedCrop([img_size], scale=(crop_min, 1.))
        self.r_diam_min=np.sqrt(crop_min)*orig_img_size
        self.orig_img_size = orig_img_size

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        lesion = 0
        if "DeepLesion" in img_path:
            slice_name = f'{img_path.split("/")[-2]}_{img_path.split("/")[-1]}'
            if slice_name in self.df.File_name.values:
                pat_row = self.df[self.df.File_name==slice_name]
                if len(pat_row)>0:
                    ks_idx = np.random.randint(0,len(pat_row))
                else:
                    ks_idx = 0
                l,b,r,t = [int(float(c)) for c in pat_row["Bounding_boxes"].values[ks_idx].split(",")]
                c_diam = np.random.randint(self.r_diam_min,self.orig_img_size,2)
                img_t = self.resize(transforms.functional.crop(img,top=t-c_diam[1]//2,left=l-c_diam[0]//2,
                                                             height=c_diam[1],width=c_diam[0]))
                img_s = self.resize(img)
                lesion = 1
            else:
                img_t = self.rrc(img)
                img_s = self.rrc(img)
        img_t = self.transform2(img_t)
        img_s = self.transform1(img_s)
        
        return [img_t, img_s], lesion