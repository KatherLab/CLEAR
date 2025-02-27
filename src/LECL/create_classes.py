#%%
import numpy as np
import os
from os.path import exists
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    
    pic_path = Path(r"/path/to/images")
    out_path = r"/path/to/outdir/"
    
    
    files = list(pic_path.glob(f"./*/*.jpg"))
    
    df = pd.read_csv("k7_with_images.csv")
    np_data = df.values

    tr_targets = np_data[np_data[:, 8] == True][:, 0].astype(int)
    vetos = np_data[np_data[:, 8] == False][:, 0].astype(int)

    used_files = [str(f) for f in files if int(str(f).split("/")[-1].split(".jpg")[0]) in tr_targets]
    if not exists(out_path):
        os.mkdir(out_path)
        os.mkdir(f"{out_path}/disease")
        os.mkdir(f"{out_path}/healthy")
    for f in used_files:
        id = f.split("/")[-1].split(".jpg")[0]
        os.system(f"cp {f} {out_path}/disease/{id}.jpg")
    for f in files:
        id = str(f).split("/")[-1].split(".jpg")[0]
        if int(id) not in vetos and int(id) not in tr_targets:
            os.system(f"cp {f} {out_path}/healthy/{id}.jpg")
# %%
