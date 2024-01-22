import json
import os


# To convert llava/llava_pretrain/images/ to llava/ as on juwels


# File path to your JSON file
path_to_data = '/p/fastdata/mmlaion/ShareGPT4V'


print("Start loading json file...")
# Read the JSON file
with open(os.path.join(path_to_data, 'captions/sharegpt4v_instruct_gpt4-vision_cap100k.json'), 'r') as file:
# with open(os.path.join(path_to_data, 'captions/share-captioner_coco_lcs_sam_1246k_1107.json'), 'r') as file:
    # for testing purpose, only load the last 1000 data
    # json_data = json.load(file)[-1000:]
    json_data = json.load(file)

count = 0

print("Start modifying json file...")
# Modify the JSON data
for item in json_data:
    image_path = item.get("image", "")


    if image_path.startswith("llava/llava_pretrain/images"):
        # Remove the prefix and replace with "llava"
        new_image_path = "llava/" + image_path[len("llava/llava_pretrain/images/"):]
        item["image"] = new_image_path

        count += 1

        # Print every 50000 items
        if count % 50000 == 0:
            print("Number of modified items: ", count)

print("Number of total modified items: ", count)



# # os.makedirs(os.path.dirname("/p/scratch/ccstdl/xu17/jz/seed_token/modified_json/captions/modified_sharegpt4v_instruct_gpt4-vision_cap100k.json"), exist_ok=True)
# os.makedirs(os.path.dirname("/p/scratch/ccstdl/xu17/jz/seed_token/modified_json/captions/modified_share-captioner_coco_lcs_sam_1246k_1107.json"), exist_ok=True)

# Write the modified data back to the JSON file
with open(os.path.join(path_to_data, '/p/scratch/ccstdl/xu17/jz/seed_token/modified_json/captions/modified_sharegpt4v_instruct_gpt4-vision_cap100k.json'), 'w') as file:
# with open(('/p/scratch/ccstdl/xu17/jz/seed_token/modified_json/captions/modified_share-captioner_coco_lcs_sam_1246k_1107.json'), 'w') as file:
    json.dump(json_data, file, indent=4)

print("JSON file has been modified successfully.")

