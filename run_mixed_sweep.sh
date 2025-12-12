#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=(0.11)
cooldown_frac=1.0
adamw_cooldown_frac=0.7
lr_scheduling="linear"
wd_mul=1.44

for muon_lr in ${muon_lr[@]}; do
    torchrun --standalone --nproc_per_node=8 train_mixed.py \
    --muon_lr ${muon_lr} \
    --cooldown_frac ${cooldown_frac} \
    --cooldown_frac_adamw ${adamw_cooldown_frac} \
    --lr_scheduling ${lr_scheduling} \
    --wd_mul ${wd_mul} \
    > logs/mixed_hybrid_norm_muonlr${muon_lr}_wd${wd_mul}.out 2> logs/mixed_hybrid_norm_muonlr${muon_lr}_wd${wd_mul}.err
done