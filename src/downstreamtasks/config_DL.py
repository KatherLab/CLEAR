import os
import json
import torch
from glob import glob
class TrainSupervisedConfig:
    def __init__(self, features, custom_config=None):
        self.features = features
        self.clini_csv = ''
        self.k_folds = 5
        # Default configuration
        self.train_data = "*_train.csv"
        self.val_data = "*_valid.csv"
        self.test_data = "*_test.csv"
        self.csv_img_key = 'image'
        self.gpu_id = "1"
        self.csv_caption_key = ['bone', 'abdomen', 'mediastinum', 'liver', 'lung', 'kidney', 'soft tissue', 'pelvis']
        
        # Set feature-specific configurations
        if features == 'BiomedCLIP_Features':
            self.indim = 512
            self.model_name = "ABMIL_Classification"
            self.num_feats = 512
        elif features in ['CTCLIP_Features', 'Merlin_Features']:
            self.indim = 512
            self.model_name = "Simple_Classification"
            self.num_feats = 1
        elif features == 'SAM_Features':
            self.indim = 256
            self.model_name = "ABMIL_Classification"
            self.num_feats = 512
        elif features == 'vit-s-ibot_Features':
            self.indim = 384
            self.model_name = "ABMIL_Classification"
            self.num_feats = 512
        else:
            self.indim = 768
            self.model_name = "ABMIL_Classification"
            self.num_feats = 512
        
        self.feat_path = "path/to/features"
        self.learning_rate = 1e-4
        self.csv_separator = ','
        self.batch_size = 128
        self.outdir = features.replace('_Features', '')
        self.base_path = f"path/to/results/{self.outdir}"
        
        # Create directories
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            if self.model_name == "ABMIL_Classification":
                os.makedirs(f'{self.base_path}/attentionplots')
        
        # Training-related configuration
        self.early_stopping = 8
        self.num_epochs = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.true_labels = {'yes': 1, 'no': 0}
        
        # Override with custom configuration if provided
        if custom_config:
            for key, value in custom_config.items():
                setattr(self, key, value)

    def save_to_json(self, file_path):
        config_dict = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
        
        with open(file_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

