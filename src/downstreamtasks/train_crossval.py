#%%
import torch
import sklearn.preprocessing
import sklearn.model_selection

import sys
from datetime import datetime
from pathlib import Path
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../")
import argparse
from utils.config_DL import TrainSupervisedConfig
from trainer import train, evaluateDL,evaluate
from model import PMA_Classification, ABMIL_Classification, Simple_Classification 
from data import FeatDatasetDL, CrossvalDatasetDL
from torch.utils.data import DataLoader
# Set up configurations
torch.manual_seed(1337)

import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run script with different configs')
    parser.add_argument('--features', type=str, help='Name of the model')

    # Parse arguments
    args = parser.parse_args()

    # Load your config
    config = TrainSupervisedConfig(features=args.features)

    # Get current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d%m%Y_%H%M%S")
    base_path = config.base_path
    #os.makedirs(base_path)
    outdim = len(config.csv_caption_key)

    if config.model_name == 'ABMIL_Classification':
        model = ABMIL_Classification(config.indim ,outdim)

    #ensure same device is used
    model=model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Training loop

    config.save_to_json(f"{config.base_path}/config.json")
    
    df = pd.read_csv(config.clini_csv)
    df['patient'] = df['image'].str.split('_').str[0]
    kfold = sklearn.model_selection.KFold(n_splits=config.k_folds, shuffle=True, random_state=1337)
    filtered_df = df.drop_duplicates('patient', keep='first')
    for fold, (train_index, test_index) in enumerate(kfold.split(filtered_df['patient'], filtered_df['all'])):
        print(f"Fold {fold + 1}/{config.k_folds}")
        if os.path.exists(f'{config.base_path}/fold-{fold}'):
            train_df = pd.read_csv(f'{base_path}/fold-{fold}/train.csv')
            valid_df = pd.read_csv(f'{base_path}/fold-{fold}/valid.csv')
        else:
            img_ids_in_fold = filtered_df['patient'].values[test_index]
            train_df = df[~df['patient'].isin(img_ids_in_fold)]
            valid_df = df[df['patient'].isin(img_ids_in_fold)]
            os.makedirs(f'{base_path}/fold-{fold}', exist_ok=True)
            train_df.to_csv(f'{base_path}/fold-{fold}/train.csv', index=False)
            valid_df.to_csv(f'{base_path}/fold-{fold}/valid.csv',index=False)
        
        im_path = f'{config.feat_path}/{config.features}'
        train_dataset = CrossvalDatasetDL(train_df, im_path, config.csv_img_key, config.csv_caption_key, num_feats = config.num_feats)
        val_dataset = CrossvalDatasetDL(valid_df, im_path, config.csv_img_key,config.csv_caption_key, num_feats = config.num_feats)
        test_dataset = FeatDatasetDL(config.test_data, im_path, config.csv_img_key,config.csv_caption_key,  num_feats = config.num_feats)
        train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last= False)
        valid_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        #ensure same device is used
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        # Training loop
        model_path = f"{base_path}/fold-{fold}/{config.model_name}_{formatted_datetime}.pth"
        odir=f"{base_path}/fold-{fold}"
        train(model, train_dl, valid_dl, optimizer, model_path, odir, config)
        evaluate(model, test_dl, optimizer, model_path, odir, config)

if __name__ == '__main__':
    main()