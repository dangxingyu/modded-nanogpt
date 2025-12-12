#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=1:59:00
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --qos=pli-cp
#SBATCH --job-name=hyperball
#SBATCH --output=logs/hyperball_%j.out
#SBATCH --error=logs/hyperball_%j.err

source ~/.bashrc
conda activate speedrun

mkdir -p logs

torchrun --standalone --nproc_per_node=8 train_origin.py