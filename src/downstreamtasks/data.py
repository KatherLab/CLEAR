import os
from dataclasses import dataclass
from multiprocessing import Value
import numpy as np
import pandas as pd
import torch
#import torchvision.datasets as datasets
import h5py


class FeatDatasetDL(Dataset):
    def __init__(self,input_filename,feat_dir, feat_key ="Feature_id",t_key="Answer", target_dict = None, num_feats=512, sep=','):
        self.feat_dir = feat_dir
        df = pd.read_csv(input_filename, sep=sep)
        self.df = filter_df(df, feat_dir, feat_key, feat=True)
        #self.feat_files = glob(f"{feat_dir}/*.h5")
        self.num_feats = num_feats
        self.t_key = t_key
        self.feat_key = feat_key
        self.target_dict = target_dict
        self.feat_files = self.df[feat_key].values
        self.keyslicedf = pd.read_csv('/path/to/DL-metadata/DL_info_KS.csv')
        self.keyslicedf['pt_id']= self.keyslicedf['File_name'].str.extract('(\w+)_(\w+)')[0]
        print(f"Found {len(self.feat_files)} feature files.")
        print(f"class distribution {self.get_num_samples()}")
        
    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        imfile = f'{self.feat_dir}/{self.feat_files[idx]}'
        feats = self.get_features(imfile)
        if self.target_dict:
            target = torch.from_numpy(np.array(self.target_dict[self.df[self.t_key].values[idx]]))
        else:
            target = torch.from_numpy(np.array(self.df[self.t_key].values[idx]))
        if 'CTCLIP' in self.feat_dir:
            feats = np.expand_dims(feats, axis=0)
        idxs = np.arange(len(feats))
        if self.num_feats > 1:
            sampled_feats, idxs = self.pad_or_sample(torch.from_numpy(feats).squeeze(1),torch.from_numpy(idxs),self.num_feats, pt = self.feat_files[idx].split('.')[0])
        else:
            sampled_feats, idxs = torch.from_numpy(feats).squeeze(1),torch.from_numpy(idxs)
        
        return sampled_feats, idxs.to(torch.float), target.to(torch.long), self.feat_files[idx] 
    def get_targets(self):
        target = np.array(self.df[self.t_key].values)
        return target
    
    def get_num_samples(self): 
        if self.target_dict:
            counts = np.unique(self.df[self.t_key].values, return_counts=True)[1] 
        else:
            counts = np.sum(self.df[self.t_key].values == 1, axis=0)
        return torch.Tensor(counts).to(torch.float32)
    
    def pad_or_sample(self, x: torch.Tensor,idxs: torch.Tensor, n: int, pt: str, deterministic: bool = False) -> torch.Tensor:
        length = x.shape[0]
        if length <= n:
            # Too few features; pad with zeros
            pad_size = n - length
            padded = torch.cat([x, torch.zeros(pad_size, x.shape[1:][0])])
            idxs_padded = torch.cat([idxs, torch.zeros(pad_size)-1])
            return padded, idxs_padded
        elif deterministic:
            # Sample equidistantly
            idx = torch.linspace(0, len(x) - 1, steps=n, dtype=torch.int)
            return x[idx], idxs[idx]
        else:
            key_slices_idx = self.keyslicedf['New_index'][self.keyslicedf['pt_id']==pt]
            t1 = torch.tensor([val*2 for val in key_slices_idx if val*2<len(idxs)])
            t2 = torch.tensor([(val*2+1) for val in key_slices_idx if val*2<len(idxs)])
            error_t1 = torch.tensor([val*2 for val in key_slices_idx if val*2>len(idxs)])
            error_t2 = torch.tensor([(val*2+1) for val in key_slices_idx if val*2>len(idxs)])
            # Sample randomly
            assert len(error_t1)==0, f'{pt} {t1} {t2} {error_t1} {error_t2}'
            used_indices = torch.cat((t1, t2))
            mask = torch.ones(length, dtype=torch.bool)
            mask[used_indices] = False

            # Get available indices and randomly permute them
            available_indices = torch.arange(length)[mask]
            perm = available_indices[torch.randperm(len(available_indices))]
            t3 = perm[:n-len(used_indices)]

            idx = torch.cat((t1, t2, t3))
            return x[torch.sort(idx)[0]], idxs[torch.sort(idx)[0]]
        
    def get_features(self, im):
        im_file = f"{im}.h5"
        try:
            with h5py.File(im_file, "r") as f:
                assert f["feats"], im_file
                feats = np.array(f["feats"][:])
                return feats
        except:
            print(im_file)


class CrossvalDatasetDL(Dataset):
    def __init__(self,df,feat_dir, feat_key ="Feature_id",t_key="Answer", target_dict = None, num_feats=512, pt=str, sep=','):
        self.df = filter_df(df, feat_dir, feat_key, feat=True)
        #self.feat_files = glob(f"{feat_dir}/*.h5")
        self.feat_dir = feat_dir
        self.num_feats = num_feats
        self.t_key = t_key
        self.feat_key = feat_key
        self.target_dict = target_dict
        self.feat_files = self.df[feat_key].values
        self.keyslicedf = pd.read_csv('/path/to/dl/metadata/DL_info_KS.csv')
        self.keyslicedf['pt_id']= self.keyslicedf['File_name'].str.extract('(\w+)_(\w+)')[0]
        print(f"Found {len(self.feat_files)} feature files.")
        print(f"class distribution {self.get_num_samples()}")
        
    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        imfile = f'{self.feat_dir}/{self.feat_files[idx]}'
        assert os.path.getsize(f'{imfile}.h5')>=1, f'error {imfile}'
        feats = self.get_features(imfile)
        if self.target_dict:
            target = torch.from_numpy(np.array(self.target_dict[self.df[self.t_key].values[idx]]))
        else:
            target = torch.from_numpy(np.array(self.df[self.t_key].values[idx]))
        if 'CTCLIP' in self.feat_dir:
            feats = np.expand_dims(feats, axis=0)
        idxs = np.arange(len(feats))
        if self.num_feats > 1:
            sampled_feats, idxs = self.pad_or_sample(torch.from_numpy(feats).squeeze(1),torch.from_numpy(idxs), self.num_feats,pt = self.feat_files[idx].split('.')[0])
        else:
            sampled_feats, idxs = torch.from_numpy(feats).squeeze(1),torch.from_numpy(idxs)
        
        return sampled_feats, idxs.to(torch.float), target.to(torch.float), self.feat_files[idx] #, target.to(torch.long), self.feat_files[idx] #
    
    def get_targets(self):
        #return self.df[self.t_key].values #this works when labels are structure in one variable
        #return self.target_dict.values()
        target = np.array(self.df[self.t_key].values)
        return target
    
    def get_num_samples(self): 
        if self.target_dict:
            counts = np.unique(self.df[self.t_key].values, return_counts=True)[1] #this works when labels are structure in one variable
        else:
            counts = np.sum(self.df[self.t_key].values == 1, axis=0)
        return torch.Tensor(counts).to(torch.float32)
    
    def pad_or_sample(self, x: torch.Tensor,idxs: torch.Tensor, n: int, pt: str, deterministic: bool = False) -> torch.Tensor:
        length = x.shape[0]
        if length <= n:
            # Too few features; pad with zeros
            pad_size = n - length
            padded = torch.cat([x, torch.zeros(pad_size, x.shape[1:][0])])
            idxs_padded = torch.cat([idxs, torch.zeros(pad_size)-1])
            return padded, idxs_padded
        elif deterministic:
            # Sample equidistantly
            idx = torch.linspace(0, len(x) - 1, steps=n, dtype=torch.int)
            return x[idx], idxs[idx]
        else:
            key_slices_idx = self.keyslicedf['New_index'][self.keyslicedf['pt_id']==pt]
            t1 = torch.tensor([val*2 for val in key_slices_idx])
            t2 = torch.tensor([(val*2+1) for val in key_slices_idx])
            # Sample randomly
            t3 = torch.randperm(length)[:n-len(t1)*2]
            idx = torch.cat((t1, t2,t3))
            return x[torch.sort(idx)[0]], idxs[torch.sort(idx)[0]]
        
    def get_features(self, im):
        im_file = f"{im}.h5"
        try:
            with h5py.File(im_file, "r") as f:
                assert f["feats"], im_file
                feats = np.array(f["feats"][:])
                return feats
        except:
            print(im_file)
