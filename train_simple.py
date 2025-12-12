import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import glob
import contextlib
from dataclasses import dataclass

import torch
torch.empty(1, device='cuda', requires_grad=True).backward()
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(nn.Module):

    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, inv_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.rotary = Rotary(dim // num_heads) # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class MoE(nn.Module):
    def __init__(self, dim, num_experts=8, num_active=2):
        super().__init__()
        self.router = CastedLinear(dim, num_experts)
        self.experts = nn.ModuleList([MLP(dim) for _ in range(num_experts)])
        self.num_active = num_active

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat)
        # top-k
        weights, indices = torch.topk(logits, self.num_active, dim=-1)
        # softmax
        weights = F.softmax(weights, dim=-1)
        
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if not mask.any():
                continue
            batch_idx, k_idx = torch.where(mask)
            expert_out = expert(x_flat[batch_idx])
            w = weights[batch_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, batch_idx, w * expert_out)
        
        return out.view(B, T, D)

class Block(nn.Module):

    def __init__(self, model_dim, num_heads, num_experts, num_active):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MoE(model_dim, num_experts=num_experts, num_active=num_active)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, num_layers, num_heads, model_dim, num_experts, num_active):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_experts, num_active) for _ in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()

    def forward(self, inputs, targets):
        x = self.embed(inputs[None]) # token embeddings of shape (b, t, model_dim)
        x = norm(x)
        
        aux_loss = 0.0
        for block in self.blocks:
            x = block(x)
            if hasattr(block.mlp, 'aux_loss'):
                aux_loss += block.mlp.aux_loss

        x = norm(x)
        logits = self.lm_head(x)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # add aux loss with a small coefficient (e.g. 0.01 or 0.1)
        return loss + 0.01 * aux_loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(path, False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with open(path, 'rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header'
    return tokens

class DistributedDataLoader:

    def __init__(self, filename_pattern):
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, batch_size):
        assert batch_size % self.world_size == 0
        device_batch_size = batch_size // self.world_size
        # load next shard if necessary
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        pos = self.current_position + self.rank * device_batch_size
        device_batch_tokens = self.tokens[pos:pos+device_batch_size+1]
        # advance current position
        self.current_position += batch_size
        inputs = device_batch_tokens[:-1].to(device='cuda', dtype=torch.int32, non_blocking=True)
        targets = device_batch_tokens[1:].to(device='cuda', dtype=torch.int64, non_blocking=True)
        return inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_bin : str = 'data/finewebedu10B/finewebedu_train_*.bin' # input .bin to train on
    val_bin : str = 'data/finewebedu10B/finewebedu_val_*.bin' # input .bin to eval validation loss on
    # optimization
    batch_size : int = 8*64*1024 # batch size in tokens
    device_batch_size : int = 64*1024 # batch size per device in tokens
    num_iterations : int = 2000 # number of iterations to run
    cooldown_iters : int = 1800 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    warmup_iters : int = 200 # number of iterations of linear warmup for triangular or trapezoidal schedule
    bf16_embeds : bool = True
    # evaluation and logging
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # implementation
    save_checkpoint : bool = False
    num_experts : int = 8
    num_active_experts : int = 2
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
assert torch.cuda.is_available()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', device_id=torch.device(local_rank))
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# adjust batch size for the number of devices
args.batch_size = args.device_batch_size * world_size
assert args.batch_size % args.device_batch_size == 0
assert (args.batch_size // args.device_batch_size) == world_size

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs('logs', exist_ok=True)
    logfile = f'logs/{run_id}.txt'
    print(logfile)

def print0(s, console=False):
    if master_process:
        with open(logfile, 'a') as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0('='*100)
# log information about the hardware/software environment this is running on
print0(f'Running Python {sys.version}')
print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}')
print0('='*100)

# load data
train_loader = DistributedDataLoader(args.train_bin)
val_loader = DistributedDataLoader(args.val_bin)
print0(f'Training dataloader files: {train_loader.files}')
print0(f'Validation dataloader files: {val_loader.files}')
print0('='*100)
inputs_train, targets_train = train_loader.next_batch(args.batch_size)

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
model = GPT(vocab_size=50304, num_layers=16, num_heads=8, model_dim=1024, 
            num_experts=args.num_experts, num_active=args.num_active_experts)
model = model.cuda()
if args.bf16_embeds:
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
model = torch.compile(model)
ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)

# init the optimizer(s)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
optimizers = [optimizer]

# learning rate decay scheduler (stable then decay)
def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return it / args.warmup_iters
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.cooldown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # --------------- VALIDATION SECTION -----------------
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        # calculate the number of steps to take in the val loop.
        assert args.val_tokens % args.batch_size == 0
        val_steps = args.val_tokens // args.batch_size
        for _ in range(val_steps):
            with torch.no_grad():
                inputs_val, targets_val = val_loader.next_batch(args.batch_size)
                val_loss += ddp_model(inputs_val, targets_val)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # logging
        print0(f'step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms', console=True)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f'logs/{run_id}', exist_ok=True)
            torch.save(log, f'logs/{run_id}/state_step{step:06d}.pt')
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    model.train()
    ddp_model(inputs_train, targets_train).backward()
    inputs_train, targets_train = train_loader.next_batch(args.batch_size)
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f'step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms', console=True)

print0(f'peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB')
dist.destroy_process_group()