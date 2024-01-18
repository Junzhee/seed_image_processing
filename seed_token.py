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

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


tokenizer_cfg_path = '/p/scratch/ccstdl/xu17/jz/seed_token/seed_llama_tokenizer_hf.yaml'
transform_cfg_path = '/p/scratch/ccstdl/xu17/jz/seed_token/clip_transform.yaml'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg, device="cuda", load_diffusion=False)

transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)


print('Loading model and tokenizer finished...')


BS = 2048
# num_tokens_per_image = 625

# Custom dataset class to handle your JSON data
class CustomDataset(Dataset):
    def __init__(self, json_data, image_root):
        self.data = [item for item in json_data if os.path.exists(os.path.join(image_root, item["image"]))]
        self.image_root = image_root
        # self.transform = transforms.Compose([
        #     transforms.Resize((200, 200)), # 200x200 -> 625 tokens
        #     transforms.ToTensor(),
        # ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = f'{self.image_root}/{self.data[idx]["image"]}'
        image = Image.open(image_path).convert('RGB')
        
        # refer to the orginal seed
        image = np.array(image.resize((256, 256))).astype(np.float32)
        image = image.transpose(2, 0, 1) / 255.0
        
        return image, idx


path = '/p/fastdata/mmlaion/ShareGPT4V'

# Load JSON data
with open(os.path.join(path, 'captions/sharegpt4v_instruct_gpt4-vision_cap100k.json'), 'r') as file:
# with open(os.path.join(path, 'captions/share-captioner_coco_lcs_sam_1246k_1107.json'), 'r') as file:
    json_data = json.load(file)

print("preview the json data: ", json_data[0])
print('Loading json data finished...')

# Define model and move it to GPU
# vqgan_model = get_movqgan_model('270M', pretrained=True, device='cpu')

# class EncoderModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vqgan_model = get_movqgan_model('270M', pretrained=True, device='cuda')

#     def forward(self, x):
#         return self.vqgan_model.encode(x)
    
# model = DataParallel(EncoderModule())
# model.to('cuda')

# Process images in batches using 4 GPUs
dataset = CustomDataset(json_data, os.path.join(path, 'data'))
data_loader = DataLoader(dataset, batch_size=BS, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4)

print("data_loader: ", data_loader)

# Process and add image tokens
for images, indices in tqdm(data_loader):
    # Split images manually and process them on each GPU
    split_images = torch.chunk(images, chunks=4)
    encoded_parts = []

    for chunk in split_images:
        chunk = chunk.to("gpu")

        print("chunk.shape: ", chunk.shape)

        with torch.no_grad():
          
            transformed_chunk = transform(chunk)
            
            chunk_tokens = tokenizer.encode_image(image_torch=transformed_chunk)
            encoded_parts.append(chunk_tokens.cpu())

            # view the first three image tokens processed
            print("chunk_tokens: ", chunk_tokens[:3])
            print("encoded_parts: ", encoded_parts[:3])

            # calculate the number of tokens per image in the chunk
            print("chunk_tokens.shape: ", chunk_tokens.shape[0])
            print("chunk.shape: ", chunk.shape[0])
            num_tokens_per_image = chunk_tokens.shape[0] // chunk.shape[0]
            print("num_tokens_per_image: ", num_tokens_per_image)

    tokens = torch.cat(encoded_parts, dim=0)

    # print the shape for checking
    print("tokens.shape: ", tokens.shape[0])
    print("expected shape: ", images.shape[0] * num_tokens_per_image)
    
    assert tokens.shape[0] == images.shape[0] * num_tokens_per_image
    # Assuming tokens are the desired output, add them to JSON
    for i, idx in enumerate(indices):
        json_data[idx]['image_token'] = tokens[i*num_tokens_per_image:(i+1)*num_tokens_per_image].cpu().numpy().tolist()

# Save the modified JSON data
with open('share-captioner_coco_lcs_sam_1246k_1107.json', 'w') as file:
    json.dump(json_data, file)
