#!/bin/bash
set -e # 遇到错误立即停止

# 创建 logs 目录
mkdir -p logs

# =================配置区域=================
# 定义要扫描的参数范围
# MUON_LRS=(0.011 0.012 0.013 0.014)
MUON_LRS=(0.0125 0.013)
VELOCITY_MOMENTUMS=(0.98)
WD_MULS=(1.56 1.69)
COOLDOWN_FRACS=(1.0)

# 固定参数
LR_SCHEDULING="linear"
GEMMA_LR=0.015
COOLDOWN_FRAC_ADAMW=0.7
# =========================================

# 遍历所有参数组合
for muon_lr in "${MUON_LRS[@]}"; do
    for velocity_momentum in "${VELOCITY_MOMENTUMS[@]}"; do
        for wd_mul in "${WD_MULS[@]}"; do
            for cooldown_frac in "${COOLDOWN_FRACS[@]}"; do
                
                # 构建日志文件名，方便识别
                LOG_TAG="muonlr${muon_lr}_vm${velocity_momentum}_wd${wd_mul}_cd${cooldown_frac}"
                LOG_FILE="logs/sweep_optimizer_only_${LOG_TAG}.out"
                
                echo "Running: $LOG_TAG"
                echo "Logs will be saved to: $LOG_FILE"
                
                # 运行训练脚本
                # 默认使用 8 卡，根据实际情况调整 --nproc_per_node
                torchrun --standalone --nproc_per_node=8 train_mixed_optimizer_only.py \
                    --muon_lr $muon_lr \
                    --velocity_momentum $velocity_momentum \
                    --wd_mul $wd_mul \
                    --cooldown_frac $cooldown_frac \
                    --lr_scheduling $LR_SCHEDULING \
                    --gemma_lr $GEMMA_LR \
                    --cooldown_frac_adamw $COOLDOWN_FRAC_ADAMW \
                    > "$LOG_FILE" 2>&1
                
                echo "Finished: $LOG_TAG"
                echo "------------------------------------------------------------------"
                
            done
        done
    done
done

