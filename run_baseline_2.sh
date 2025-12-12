#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=0.013
cooldown_frac=1.0
adamw_cooldown_frac=0.6
lr_scheduling="linear"

torchrun --standalone --nproc_per_node=8 train_baseline_2.py \
    --muon_lr ${muon_lr} \
    --cooldown_frac ${cooldown_frac} \
    --cooldown_frac_adamw ${adamw_cooldown_frac} \
    --lr_scheduling ${lr_scheduling} \
    > logs/baseline_constant_momentum_muonlr${muon_lr}_cooldown${cooldown_frac}_adamwcooldown${adamw_cooldown_frac}_${lr_scheduling}.out 2> logs/baseline_constant_momentum_muonlr${muon_lr}_cooldown${cooldown_frac}_adamwcooldown${adamw_cooldown_frac}_${lr_scheduling}.err