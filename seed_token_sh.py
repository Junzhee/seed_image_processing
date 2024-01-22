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

import argparse


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

# Custom dataset class to handle your JSON data
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


def process_chunk(rank, world_size, json_data, path, tokenizer_cfg_path, transform_cfg_path, batch_size, output_dir, num_workers, prefetch_factor):
    
    torch.cuda.set_device(rank)
    gpu_name = torch.cuda.get_device_name(rank)
    print(f"Process {rank} is using GPU: {gpu_name}")
    print("time: ", datetime.datetime.now())
    start_time_on_device = datetime.datetime.now()

    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=rank, load_diffusion=False)

    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    print(f'Loading model and tokenizer finished on device {rank}...')
    print("time: ", datetime.datetime.now())


    start_idx = rank * len(json_data) // world_size
    end_idx = (rank + 1) * len(json_data) // world_size

    dataset = CustomDataset(json_data[start_idx:end_idx], path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)

    try:
        print(f"Image Processing start on device {rank}, time: ", datetime.datetime.now())

        for images, indices in tqdm(data_loader, position=rank):
            images = images.to(f'cuda:{rank}')

            with torch.no_grad():
                transformed_images = transform(images)
                tokens = tokenizer.encode_image(image_torch=transformed_images)
                
                # print(f"Debug: Tokens on device {rank} - {tokens[:3]}")  

            updated_data = []

            for i, idx in enumerate(indices):
                actual_idx = start_idx + idx
                
                # print(f"Debug: Updating index {actual_idx} on device {rank}")  
    
                json_data[actual_idx]['image_token'] = tokens[i].cpu().numpy().tolist()
                updated_data.append(json_data[actual_idx])

                # print(f"Debug: Updated data at index {actual_idx} on device {rank} - {json_data[actual_idx]}")

            output_filename = os.path.join(output_dir, f'processed_data_chunk_{rank}.json')
            with open(output_filename, 'w') as file: 
                json.dump(json_data[start_idx:end_idx], file)

        print(f"Image Ppocessing finished on device {rank}, time: ", datetime.datetime.now())
        # print the total time of the current process
        print(f"Total time consumed on device {rank}: ", datetime.datetime.now() - start_time_on_device)

       

        # view some of the processed data
        # print(f"sample processed data on device {rank}: ", json_data[start_idx:start_idx+3])

    except Exception as e:
        logging.error("An error occurred: ", exc_info=True)


def combine_chunks(output_dir):
    combined_data = []
    for file_name in os.listdir(output_dir):
        if file_name.startswith('processed_data_chunk_') and file_name.endswith('.json'):
            with open(os.path.join(output_dir, file_name), 'r') as file:
                data = json.load(file)
                combined_data.extend(data)

    with open(os.path.join(output_dir, 'combined_data.json'), 'w') as file:
        json.dump(combined_data, file)

    print("Combined all chunks into a single file.")

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_gpus', type=int, required=True, default=4, help='Number of GPUs')
    parser.add_argument('--BS', type=int, required=True, help='Batch Size')
    parser.add_argument('--path_to_data', type=str, required=True, help='Path to Data')
    parser.add_argument('--tokenizer_cfg_path', type=str, required=True, help='Tokenizer Config Path')
    parser.add_argument('--transform_cfg_path', type=str, required=True, help='Transform Config Path')
    parser.add_argument('--path_to_json', type=str, required=True, help='Path to JSON')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of workers of dataloader')
    parser.add_argument('--prefetch_factor', type=int, required=True, help='Prefetch factor of dataloader')
    
    return parser.parse_args()


def main():


    args = parse_args()
    num_gpus = args.num_gpus
    BS = args.BS
    path_to_data = args.path_to_data
    output_dir = args.output_dir
    tokenizer_cfg_path = args.tokenizer_cfg_path
    transform_cfg_path = args.transform_cfg_path
    path_to_json = args.path_to_json
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor

    # print the settings:
    print(f"Number of GPUs: {num_gpus}")
    print(f"Batch Size: {BS}")
    print(f"Path to Data: {path_to_data}")
    print(f"Output Directory: {output_dir}")
    print(f"Tokenizer Config Path: {tokenizer_cfg_path}")
    print(f"Transform Config Path: {transform_cfg_path}")
    print(f"Path to JSON: {path_to_json}")
    print(f"Number of workers of dataloader: {num_workers}")
    print(f"Prefetch factor of dataloader: {prefetch_factor}")
    

    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_filename = f"log_{current_time}.txt"

    # logging.basicConfig(filename=log_filename, level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    start_time = datetime.datetime.now()
    print(f"All process start time: {start_time}")  
    

    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    with open(os.path.join(path_to_data, path_to_json), 'r') as file:
        # json_data = json.load(file)[:1000]
        json_data = json.load(file)



    mp.spawn(process_chunk,
            args=(num_gpus, json_data, os.path.join(path_to_data, 'data'), tokenizer_cfg_path, transform_cfg_path, BS, output_di, num_workers, prefetch_factor),
            nprocs=num_gpus,
            join=True)

    print("All processes finished, time: ", datetime.datetime.now())
    print("Processing time consumed: ", datetime.datetime.now() - start_time)

    combine_chunks(output_dir)


    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"Total time consumed: {total_time}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
