#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

torchrun --standalone --nproc_per_node=8 train_gpt_numerical.py > logs/weird_lr_unified.out 2> logs/weird_lr_unified.err