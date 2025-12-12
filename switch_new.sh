#!/bin/bash

# Hyperparameter sweep script for train_baseline.py
# Usage: ./sweep_baseline.sh

source ~/.bashrc
conda activate speedrun

mkdir -p logs


muon_lr=0.025
cooldown_frac=0.6
phase_2_ratio=0.24

torchrun --standalone --nproc_per_node=8 train_switch_new.py \
    --muon_lr ${muon_lr} \
    --cooldown_frac ${cooldown_frac} \
    --phase_2_ratio ${phase_2_ratio} \
    > logs/switch_new_muonlr${muon_lr}_cooldown${cooldown_frac}_phase2ratio${phase_2_ratio}.out 2> logs/switch_new_muonlr${muon_lr}_cooldown${cooldown_frac}_phase2ratio${phase_2_ratio}.err