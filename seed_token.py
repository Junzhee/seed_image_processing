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


def process_chunk(rank, world_size, json_data, path, tokenizer_cfg_path, transform_cfg_path, batch_size, output_dir):
    
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

    print("Process {} start_idx: {}, end_idx: {}".format(rank, start_idx, end_idx))

    # print the dataset consumed time
    dataset_start_time = datetime.datetime.now()
    print(f"Dataset start on device {rank}, time: ", datetime.datetime.now())
    dataset = CustomDataset(json_data[start_idx:end_idx], path)

    print(f"Dataset completed on device {rank}..., time consumed: ", datetime.datetime.now() - dataset_start_time)
    print(f"Dataset length on device {rank}: ", len(dataset))

    # print the data loader consumed time
    print(f"Data loader start on device {rank}, time: ", datetime.datetime.now())
    dataloader_start_time = datetime.datetime.now()

    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=12, 
                             pin_memory=True, 
                             persistent_workers=True,
                             prefetch_factor=4)

    print(f"Data loader finished on device {rank}, time: ", datetime.datetime.now())
    print(f"Total time consumed on device {rank}: ", datetime.datetime.now() - dataloader_start_time)
    print(f"Total number of batches on device {rank}: ", len(data_loader))



    try:
        print(f"Image Processing start on device {rank}, time: ", datetime.datetime.now())

        for images, indices in tqdm(data_loader, position=rank):
            images = images.to(f'cuda:{rank}', non_blocking = True)

            # print(f"Rank {rank} is processing the following indices: {indices}...")
            print(f"Rank {rank} is processing images with shape: {images.shape}...")
            indice_start_time = datetime.datetime.now()

            with torch.no_grad():
                transformed_images = transform(images)
                tokens = tokenizer.encode_image(image_torch=transformed_images)

                
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
                
                # print(f"Dumped data to file on device {rank}, time: ", datetime.datetime.now())


            print(f"Rank {rank} finished processing the indices:... Time consumed: ", datetime.datetime.now() - indice_start_time)
     
            


        print(f"Image Processing finished on device {rank}, time: ", datetime.datetime.now())
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

    last_dir_name = os.path.basename(output_dir)
    custom_filename = f'processed_{last_dir_name}.json'
    full_path = os.path.join(output_dir, custom_filename)


    with open(full_path, 'w') as file:
        json.dump(combined_data, file)

    # with open(os.path.join(output_dir, 'combined_data.json'), 'w') as file:
    #     json.dump(combined_data, file)

    print("Combined all chunks into a single file.")


def main():

    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_filename = f"log_{current_time}.txt"

    # logging.basicConfig(filename=log_filename, level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    start_time = datetime.datetime.now()
    print(f"All process start time: {start_time}")  
    
    num_gpus = 4 
    BS = 2048  

    path_to_data = '/p/fastdata/mmlaion/ShareGPT4V'
    path_to_json = '/p/scratch/ccstdl/xu17/jz/seed_token/modified_json/'
    
    # output_dir = '/p/scratch/ccstdl/xu17/jz/seed_token/output/sharegpt4v_instruct_gpt4-vision_cap100k'
    output_dir = '/p/scratch/ccstdl/xu17/jz/seed_token/output/share-captioner_coco_lcs_sam_1246k_1107'
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    # with open(os.path.join(path_to_json, 'captions/modified_sharegpt4v_instruct_gpt4-vision_cap100k.json'), 'r') as file:
    with open(os.path.join(path_to_json, 'captions/modified_share-captioner_coco_lcs_sam_1246k_1107.json'), 'r') as file:
        # for testing purpose, only load the first 1000 data
        # json_data = json.load(file)[:1000]
        json_data = json.load(file)

    tokenizer_cfg_path = '/p/scratch/ccstdl/xu17/jz/seed_token/model_config/seed_llama_tokenizer_hf.yaml'
    transform_cfg_path = '/p/scratch/ccstdl/xu17/jz/seed_token/model_config/clip_transform.yaml'


    mp.spawn(process_chunk,
            args=(num_gpus, json_data, os.path.join(path_to_data, 'data'), tokenizer_cfg_path, transform_cfg_path, BS, output_dir),
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
