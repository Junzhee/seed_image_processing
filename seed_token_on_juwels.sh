#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=seed_token
#SBATCH --account=cstdl
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                    # numbe of GPUs
#SBATCH --ntasks-per-node=1             # Crucial - only one task per dist per node !
#SBATCH --cpus-per-task=48           # Slurm 22.05: srun doesnot inherit this variable from sbatch
#SBATCH --time=02:00:00                 # maximum execution time (HH:MM:SS)
#SBATCH --threads-per-core=1            # using only real cores, no SMT
#SBATCH --hint=nomultithread
#SBATCH --output=./log/%x-%j.out              # output file name


#### Comments ####
# Use `develbooster` for debugging, `booster` for "normal" jobs, and
# `largebooster` for jobs on more than 256 nodes.

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Enable logging
set -euo pipefail
set -x


INSTALLATION_DIR="/p/scratch/ccstdl/xu17/jz/seed_token"
cd $INSTALLATION_DIR


source /p/scratch/ccstdl/xu17/miniconda3/bin/activate /p/scratch/ccstdl/xu17/miniconda3/envs/jz


output_dir = '/p/scratch/ccstdl/xu17/jz/seed_token/output/captions/sharegpt4v_instruct_gpt4-vision_cap100k'
# output_dir = '/p/scratch/ccstdl/xu17/jz/seed_token/output/share-captioner_coco_lcs_sam_1246k_1107'
num_gpus = 4 
BS = 2048 
path_to_data = '/p/fastdata/mmlaion/ShareGPT4V'
tokenizer_cfg_path = '/p/scratch/ccstdl/xu17/jz/seed_token/seed_llama_tokenizer_hf.yaml'
transform_cfg_path = '/p/scratch/ccstdl/xu17/jz/seed_token/clip_transform.yaml' 
path_to_json = 'captions/sharegpt4v_instruct_gpt4-vision_cap100k.json'
# path_to_json ='captions/share-captioner_coco_lcs_sam_1246k_1107.json'
num_workers = 32
prefetch_factor = 4


python seed_token_sh.py --output_dir ${output_dir} --num_gpus ${num_gpus} --BS ${BS} \
                    --path_to_data ${path_to_data} --tokenizer_cfg_path ${tokenizer_cfg_path} \
                    --transform_cfg_path ${transform_cfg_path} --path_to_json ${path_to_json} \
                    --num_workers ${num_workers} --prefetch_factor ${prefetch_factor} \

