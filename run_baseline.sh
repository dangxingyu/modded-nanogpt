#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=0.013
cooldown_frac=1.0
adamw_cooldown_frac=0.6
lr_scheduling="linear"
wd_mul=(1.21 1.69)

for wd in "${wd_mul[@]}"; do
    torchrun --standalone --nproc_per_node=8 train_baseline.py \
        --muon_lr ${muon_lr} \
        --cooldown_frac ${cooldown_frac} \
        --cooldown_frac_adamw ${adamw_cooldown_frac} \
        --lr_scheduling ${lr_scheduling} \
        --wd_mul ${wd} \
        > logs/decoupled_rescaled_${wd}_baseline_muonlr${muon_lr}_cooldown${cooldown_frac}_adamwcooldown${adamw_cooldown_frac}_${lr_scheduling}.out 2> logs/decoupled_rescaled_${wd}_baseline_muonlr${muon_lr}_cooldown${cooldown_frac}_adamwcooldown${adamw_cooldown_frac}_${lr_scheduling}.err
done