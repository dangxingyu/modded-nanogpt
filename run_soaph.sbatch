#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --qos=pli-short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xd7812@princeton.edu

source ~/.bashrc
conda activate speedrun

mkdir -p logs

soaph_lrs=(0.008 0.010 0.012 0.015 0.020)
# soaph_lrs=(0.012)
# cooldown_fracs=(1.0 0.9 0.8)
wd_muls=(1.0 1.44)
# wd_muls=(1.44)

for soaph_lr in ${soaph_lrs[@]}; do
    for wd_mul in ${wd_muls[@]}; do
        echo "Running soaph_lr=${soaph_lr}, wd_mul=${wd_mul}"
        torchrun --standalone --nproc_per_node=8 train_soaph.py \
            --soaph_lr ${soaph_lr} \
            --wd_mul ${wd_mul} \
            > logs/dec_5_soaph_lr${soaph_lr}_wd${wd_mul}.out 2> logs/dec_5_soaph_lr${soaph_lr}_wd${wd_mul}.err
    done
done

echo "Done"

