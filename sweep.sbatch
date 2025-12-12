#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --qos=pli-short
#SBATCH --job-name=last_sweep
#SBATCH --output=logs/last_sweep_%j.out
#SBATCH --error=logs/last_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xd7812@princeton.edu

source ~/.bashrc
conda activate speedrun
bash last_sweep.sh