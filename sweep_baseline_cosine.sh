#!/bin/bash

# Hyperparameter sweep script for train_baseline.py
# Usage: ./sweep_baseline.sh

source ~/.bashrc
conda activate speedrun

mkdir -p logs

# Define sweep ranges
PEAK_LRS=(0.008 0.012 0.016 0.02)  # muon_lr values
DECAY_FRACS=(1.0 0.8 0.7 0.6 0.4)      # cooldown_frac values
SCHEDULE="cosine"                   # lr_scheduling

# Run sweep
for peak_lr in "${PEAK_LRS[@]}"; do
    for decay_frac in "${DECAY_FRACS[@]}"; do
        echo "Running with muon_lr=${peak_lr}, cooldown_frac=${decay_frac}, lr_scheduling=${SCHEDULE}"

        # Create a unique run identifier
        RUN_NAME="baseline_muonlr${peak_lr}_cooldown${decay_frac}_${SCHEDULE}"

        # Run training
        torchrun --standalone --nproc_per_node=8 train_baseline.py \
            --muon_lr ${peak_lr} \
            --cooldown_frac ${decay_frac} \
            --lr_scheduling ${SCHEDULE} \
            > logs/${RUN_NAME}.out 2> logs/${RUN_NAME}.err

        echo "Completed ${RUN_NAME}"
    done
done

echo "Sweep completed!"

