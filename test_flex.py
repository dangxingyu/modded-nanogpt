import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

print(f"PyTorch version: {torch.version.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"FlexAttention available: {flex_attention is not None}")

# Test compiled_autograd
print(f"compiled_autograd before: {torch._dynamo.config.compiled_autograd}")
torch._dynamo.config.compiled_autograd = True
print(f"compiled_autograd after: {torch._dynamo.config.compiled_autograd}")

# Simple test
torch.cuda.set_device(0)
q = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.bfloat16)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = BlockMask.from_seqlens([128], [128], mask_mod=causal_mask)
print("Created block_mask successfully")

try:
    out = flex_attention(q, k, v, block_mask=block_mask)
    print("FlexAttention forward pass successful!")
    print(f"Output shape: {out.shape}")
except Exception as e:
    print(f"Error during flex_attention: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")
