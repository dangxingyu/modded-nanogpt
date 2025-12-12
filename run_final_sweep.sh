#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=(0.010 0.011 0.012 0.013)
cooldown_frac=1.0
adamw_cooldown_frac=0.7
lr_scheduling="linear"
wd_mul=(1.2 1.3 1.4 1.5)
ema_decay=(0.85)

for muon_lr in "${muon_lr[@]}"; do
    for wd in "${wd_mul[@]}"; do
        torchrun --standalone --nproc_per_node=8 train_baseline_ema.py \
            --muon_lr "${muon_lr}" \
            --cooldown_frac "${cooldown_frac}" \
            --cooldown_frac_adamw "${adamw_cooldown_frac}" \
            --lr_scheduling "${lr_scheduling}" \
            --wd_mul "${wd}" \
            --ema_decay "${ema_decay}" \
            > logs/ema_muonlr${muon_lr}_wd${wd}_ema${ema_decay}.out 2> logs/ema_muonlr${muon_lr}_wd${wd}_ema${ema_decay}.err
    done
done
