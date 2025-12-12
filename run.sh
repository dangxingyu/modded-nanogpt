export TORCHDYNAMO_VERBOSE=1
torchrun --standalone --nproc_per_node=8 train_origin_hybrid_norm.py > logs/origin_hybrid_norm.out 2> logs/origin_hybrid_norm.err