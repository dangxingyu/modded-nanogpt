#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --qos=pli-short

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=0.012
cooldown_frac=1.0
adamw_cooldown_frac=0.7
lr_scheduling="linear"
wd_mul=1.44
velocity_momentum=0.98
gemma_lrs=(0.015 0.01 0.018 0.012)

for gemma_lr in ${gemma_lrs[@]}; do
    torchrun --standalone --nproc_per_node=8 train_mixed.py \
    --muon_lr ${muon_lr} \
    --cooldown_frac ${cooldown_frac} \
    --cooldown_frac_adamw ${adamw_cooldown_frac} \
    --lr_scheduling ${lr_scheduling} \
    --wd_mul ${wd_mul} \
    --velocity_momentum ${velocity_momentum} \
    --gemma_lr ${gemma_lr} \
    > logs/mixed_muonlr${muon_lr}_wd${wd_mul}_velocitymomentum${velocity_momentum}_gemma${gemma_lr}.out 2> logs/mixed_muonlr${muon_lr}_wd${wd_mul}_velocitymomentum${velocity_momentum}_gemma${gemma_lr}.err
done