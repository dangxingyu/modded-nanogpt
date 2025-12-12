#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=0.013
cooldown_frac=1.0
adamw_cooldown_frac=0.6
lr_scheduling="linear"
wd_mul=1.44
ema_decay=0.95

torchrun --standalone --nproc_per_node=8 train_baseline_ema.py \
    --muon_lr ${muon_lr} \
    --cooldown_frac ${cooldown_frac} \
    --cooldown_frac_adamw ${adamw_cooldown_frac} \
    --lr_scheduling ${lr_scheduling} \
    --wd_mul ${wd_mul} \
    --ema_decay ${ema_decay} \
    > logs/baseline_ema${ema_decay}_muonlr${muon_lr}_wd${wd_mul}.out 2> logs/baseline_ema${ema_decay}_muonlr${muon_lr}_wd${wd_mul}.err