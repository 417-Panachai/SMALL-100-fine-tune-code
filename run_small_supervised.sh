#!/bin/bash
#SBATCH -p gpu                # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 4             # Specify number of nodes and processors per task
#SBATCH --gpus=4              # Specify total number of GPUs
#SBATCH --ntasks-per-node=1   # Specify tasks per node
#SBATCH -t 58:00:00           # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200324           # Specify project name
#SBATCH -J S-100              # Specify job name
#SBATCH -o slurm-out/S-100.out

ml Mamba
conda deactivate
conda activate /path/env

export WANDB_MODE="offline"
export WANDB_API_KEY="YOUR_API_KEY"
export HF_DATASETS_CACHE=
export TRANSFORMERS_CACHE=
export CACHE_DIR=
export HF_HOME=


python train_small_supervised.py \
    --src_to_trg_data_csv_path "./path/dir/th_en_dataset" \
    --trg_to_src_data_csv_path "./path/dir/en_th_dataset" \
    --eval_data_csv_path "./path/dir/eval_dataset" \
    --checkpoint_dir "./model/path" \
    --output_dir "./path/dir/checkpoint" \
    --num_processors 8 \
    --num_train_epochs 6 \
    --max_save_checkpoint 3 \
    --save_strategy "epoch" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_pin_memory \
    --logging_steps 100 \
    --fp16 True \
    --report_to "wandb" \
