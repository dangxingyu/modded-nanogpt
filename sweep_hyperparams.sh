#!/bin/bash

# Hyperparameter sweep script for train_gpt_numerical.py
# Usage: ./sweep_hyperparams.sh

source ~/.bashrc
conda activate speedrun

mkdir -p logs

# Define sweep ranges
ZERO_STEP_LRS=(0.03)
MUON_LRS=(0.006 0.012)
COOLDOWN_FRACS=(0.8)

# Run sweep
for zero_lr in "${ZERO_STEP_LRS[@]}"; do
    for muon_lr in "${MUON_LRS[@]}"; do
        for cooldown in "${COOLDOWN_FRACS[@]}"; do
            echo "Running with zero_step_lr=${zero_lr}, muon_lr=${muon_lr}, cooldown_frac=${cooldown}"

            # Create a unique run identifier
            RUN_NAME="renorm-zslr${zero_lr}_mlr${muon_lr}_cd${cooldown}"

            # Run training
            torchrun --standalone --nproc_per_node=8 train_gpt_numerical.py \
                --zero_step_lr ${zero_lr} \
                --muon_lr ${muon_lr} \
                --cooldown_frac ${cooldown} \
                > logs/${RUN_NAME}.out 2> logs/${RUN_NAME}.err

            echo "Completed ${RUN_NAME}"
        done
    done
done

echo "Sweep completed!"
