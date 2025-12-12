#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=1:59:00
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --qos=pli-short
#SBATCH --job-name=hyperball-numerical
#SBATCH --output=logs/hyperball-numerical_%j.out
#SBATCH --error=logs/hyperball-numerical_%j.err

source ~/.bashrc
conda activate speedrun

mkdir -p logs

torchrun --standalone --nproc_per_node=8 train_gpt_numerical.py


cooldown: 0.8 0.7
muon_lr: 0.004 0.006 0.012 0.018
zero_step_lr: 0.03 0.06