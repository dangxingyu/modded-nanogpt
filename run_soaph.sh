#!/bin/bash

# 手动运行脚本，无需 #SBATCH 指令，假设你已通过 srun/salloc 占用到8卡节点

source ~/.bashrc
conda activate speedrun

mkdir -p logs

soaph_lrs=(0.010 0.012 0.015 0.018 0.020 0.025)
# cooldown_fracs=(1.0 0.9 0.8)
wd_muls=(1.0 1.44)

for soaph_lr in ${soaph_lrs[@]}; do
    for wd_mul in ${wd_muls[@]}; do
        echo "Running soaph_lr=${soaph_lr}, wd_mul=${wd_mul}"
        torchrun --standalone --nproc_per_node=8 train_soaph.py \
            --soaph_lr ${soaph_lr} \
            --wd_mul ${wd_mul} \
            > logs/dec_3_soaph_lr${soaph_lr}_wd${wd_mul}.out 2> logs/dec_3_soaph_lr${soaph_lr}_wd${wd_mul}.err
    done
done

echo "Done"

