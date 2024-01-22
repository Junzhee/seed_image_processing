from tqdm import tqdm
import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
import pyrootutils
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch.multiprocessing as mp
import json

import logging
import datetime


class CustomDataset(Dataset):
    def __init__(self, json_data, image_root):
        self.data = [item for item in json_data if os.path.exists(os.path.join(image_root, item["image"]))]
        self.image_root = image_root
     

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = f'{self.image_root}/{self.data[idx]["image"]}'
        image = Image.open(image_path).convert('RGB')
        
        image = np.array(image.resize((256, 256))).astype(np.float32)
        image = image.transpose(2, 0, 1) / 255.0

        return image, idx

    




path_to_data = '/p/fastdata/mmlaion/ShareGPT4V'
    
path_to_json = '/p/scratch/ccstdl/xu17/jz/seed_token/modified_json/'

path = os.path.join(path_to_data, 'data')

print("Start loading json file...")

 # with open(os.path.join(path_to_data, 'captions/modified_sharegpt4v_instruct_gpt4-vision_cap100k.json'), 'r') as file:
with open(os.path.join(path_to_json, 'captions/modified_share-captioner_coco_lcs_sam_1246k_1107.json'), 'r') as file:
    # for testing purpose, only load the first 1000 data
    # json_data = json.load(file)[:1000]
    json_data = json.load(file)

print("Start loading dataset...")

split_size = len(json_data) // 4
datasets = []


for i in range(4):
    print(f"Loading dataset {i+1}...")
    start_idx = i * split_size
    end_idx = None if i == 3 else start_idx + split_size

    print(f"Index range on {i+1}th dataset: ", start_idx, end_idx)


    subset = json_data[start_idx:end_idx]

    print(f"Subset length on {i+1}th dataset: ", len(subset))

    # valid_image_count = sum(os.path.exists(os.path.join(path, item["image"])) for item in subset)
    # print(f"Number of valid images in dataset {i+1}: {valid_image_count}")

    # if i==3:
    #     print(subset)


    dataset = CustomDataset(subset, path)
    datasets.append(dataset)
    
    print(f"Finished loading dataset {i+1}...")

# Verifying the number of images in each dataset
for i, dataset in enumerate(datasets):
    print(f"Number of images in dataset {i+1}: {len(dataset)}")

print("Start loading dataloader...")

# 


# data_loader = DataLoader(dataset, 
#                             batch_size=2048, 
#                             shuffle=False, 
#                             num_workers=16, 
#                             pin_memory=True, 
#                             persistent_workers=True,
#                             prefetch_factor=4)
