#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

muon_lr=(0.012 0.011 0.013)
cooldown_fracs=(1.0 0.9 0.8)
adamw_cooldown_frac=0.7
# lr_scheduling="linear"
lr_scheduling="cosine"
wd_mul=1.44
velocity_momentums=(0.98)
gemma_lr=0.015

for muon_lr in ${muon_lr[@]}; do
    for velocity_momentum in ${velocity_momentums[@]}; do
        for cooldown_frac in ${cooldown_fracs[@]}; do
            echo "Running muon_lr=${muon_lr}, velocity_momentum=${velocity_momentum}, cooldown_frac=${cooldown_frac}, lr_scheduling=${lr_scheduling}"
            torchrun --standalone --nproc_per_node=8 train_mixed.py \
                --muon_lr ${muon_lr} \
                --cooldown_frac ${cooldown_frac} \
                --cooldown_frac_adamw ${adamw_cooldown_frac} \
                --lr_scheduling ${lr_scheduling} \
                --wd_mul ${wd_mul} \
                --velocity_momentum ${velocity_momentum} \
                --gemma_lr ${gemma_lr} \
                > logs/nov_30_mixed_muonlr${muon_lr}_wd${wd_mul}_velocitymomentum${velocity_momentum}_gemma${gemma_lr}_${lr_scheduling}_cooldownfrac${cooldown_frac}.out 2> logs/nov_30_mixed_muonlr${muon_lr}_wd${wd_mul}_velocitymomentum${velocity_momentum}_gemma${gemma_lr}_${lr_scheduling}_cooldownfrac${cooldown_frac}.err
        done
    done
done

echo "Done"