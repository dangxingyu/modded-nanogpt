import re
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import numpy as np
import glob

# File paths
log_dir = Path(__file__).parent / 'logs'
output_path = Path(__file__).parent / 'figure' / 'training_comparison_final.png'
record_file = Path(__file__).parent / 'records' / 'track_2_medium' / '2025-06-15_OptimizationLeaderboard' / '075_640429f2-e726-4e83-aa27-684626239ffc.txt'

runs = []

# --- 1. Final Files (Viridis colormap) ---
final_files = sorted(glob.glob(str(log_dir / 'final_*.out')))
final_files = [Path(f).name for f in final_files]

if final_files:
    colors = cm.viridis(np.linspace(0, 1, len(final_files)))
    for i, filename in enumerate(final_files):
        runs.append({
            'name': f'Final {i+1}',
            'file': filename,
            'filepath': log_dir / filename,
            'train_color': colors[i],
            'val_color': colors[i],
            'marker': 'o',
            'is_final': True,
            'linewidth': 2.5,
        })

# --- 2. Record Baseline (Black) ---
runs.append({
    'name': 'record-8',
    'file': '075_640429f2-e726-4e83-aa27-684626239ffc.txt',
    'filepath': record_file,
    'train_color': 'black',
    'val_color': 'black',
    'marker': 'H',
    'is_final': False,
    'linewidth': 3.0,
})

# --- 3. Mixed Files Parsing Logic ---
# Helper to safely get color and marker
def get_style(idx, total, cmap_name='tab10'):
    cmap = plt.get_cmap(cmap_name)
    # If total is small, pick distinct colors; otherwise sample evenly
    if total <= 10:
        color = cmap(idx % 10)
    else:
        color = cmap(idx / max(total - 1, 1))
    
    markers = ['v', 's', '^', 'd', '>', '<', '*', 'P', 'X', '8', 'p', 'h']
    marker = markers[idx % len(markers)]
    return color, marker

# Group A: Standard Mixed (muonlr + wd)
mixed_files = sorted(glob.glob(str(log_dir / 'mixed_muonlr*_wd*.out')))
# Filter out those that might match more specific patterns below to avoid duplication
mixed_files = [f for f in mixed_files if 'velocitymomentum' not in f and 'gemma' not in f]

for i, path in enumerate(mixed_files):
    filename = Path(path).name
    # Regex for: mixed_muonlr0.01_wd0.01.out
    match = re.search(r'muonlr([0-9.]+)_wd([0-9.]+)', filename)
    if match:
        muonlr = match.group(1)
        wd = match.group(2)
        name = f"Mixed LR{muonlr} WD{wd}"
        color, marker = get_style(i, len(mixed_files), 'tab10') # Use tab10 for this group
        runs.append({
            'name': name,
            'file': filename,
            'filepath': log_dir / filename,
            'train_color': color,
            'val_color': color,
            'marker': marker,
            'is_final': False,
            'linewidth': 2.0,
        })

# Group B: Mixed with Velocity & Gemma (The complex one)
last_mixed = sorted(glob.glob(str(log_dir / 'mixed_muonlr*_wd*_velocitymomentum*_gemma*.out')))

# 生成一组区分度高的颜色 (例如使用 nipy_spectral 或 jet)
b_colors = cm.nipy_spectral(np.linspace(0.1, 0.9, max(len(last_mixed), 1)))

for i, path in enumerate(last_mixed):
    filename = Path(path).name
    match = re.search(r'muonlr([0-9.]+)_wd([0-9.]+)_velocitymomentum([0-9.]+)_gemma([0-9.]+)', filename)
    if match:
        muonlr = match.group(1)
        wd = match.group(2)
        vm = match.group(3)
        gemma = match.group(4)
        name = f"AdaMuon LR{muonlr} M{vm} G{gemma}"
        
        # 使用单独生成的颜色列表
        color = b_colors[i % len(b_colors)]
        
        # 轮换 marker
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        marker = markers[i % len(markers)]
        
        runs.append({
            'name': name,
            'file': filename,
            'filepath': log_dir / filename,
            'train_color': color,
            'val_color': color,
            'marker': marker,
            'is_final': False,
            'linewidth': 2.5,
        })

# Group C: Mixed without Gemma
# mixed_wo_gemma = sorted(glob.glob(str(log_dir / 'mixed_w_o_gemma_muonlr*_wd*.out')))
# for i, path in enumerate(mixed_wo_gemma):
#     filename = Path(path).name
#     match = re.search(r'muonlr([0-9.]+)_wd([0-9.]+)', filename)
#     if match:
#         muonlr = match.group(1)
#         wd = match.group(2)
#         name = f"NoGemma LR{muonlr} WD{wd}"
#         color, marker = get_style(i, len(mixed_wo_gemma), 'Set2') # Use Set2 for this group
#         runs.append({
#             'name': name,
#             'file': filename,
#             'filepath': log_dir / filename,
#             'train_color': color,
#             'val_color': color,
#             'marker': marker,
#             'is_final': False,
#             'linewidth': 2.0,
#         })

# Group C: End Mixed Files (e.g. end_mixed_muonlr0.0115_wd1.44_velocitymomentum0.98_gemma0.015.out)
end_mixed = sorted(glob.glob(str(log_dir / 'end_mixed_muonlr*_wd*_velocitymomentum*_gemma*.out')))

if end_mixed:
    c_colors = cm.Set1(np.linspace(0, 1, max(len(end_mixed), 1)))
    
    for i, path in enumerate(end_mixed):
        filename = Path(path).name
        match = re.search(r'muonlr([0-9.]+)_wd([0-9.]+)_velocitymomentum([0-9.]+)_gemma([0-9.]+)', filename)
        if match:
            muonlr = match.group(1)
            wd = match.group(2)
            vm = match.group(3)
            gemma = match.group(4)
            name = f"End LR{muonlr} VM{vm} G{gemma}"
            
            color = c_colors[i % len(c_colors)]
            markers = ['X', 'P', 'D', 'd', 's', 'p', '*']
            marker = markers[i % len(markers)]
            
            runs.append({
                'name': name,
                'file': filename,
                'filepath': log_dir / filename,
                'train_color': color,
                'val_color': color,
                'marker': marker,
                'is_final': False,
                'linewidth': 2.5,
            })

# Group D: Optimizer Only Sweeps
opt_only_files = sorted(glob.glob(str(log_dir / 'sweep_optimizer_only_*.out')))

if opt_only_files:
    d_colors = cm.tab20(np.linspace(0, 1, max(len(opt_only_files), 1)))
    
    for i, path in enumerate(opt_only_files):
        filename = Path(path).name
        match = re.search(r'muonlr([0-9.]+)_vm([0-9.]+)_wd([0-9.]+)_cd([0-9.]+)', filename)
        if match:
            muonlr = match.group(1)
            vm = match.group(2)
            wd = match.group(3)
            cd = match.group(4)
            name = f"optimizer-only AdaMuon LR{muonlr} VM{vm} WD{wd}"
            
            color = d_colors[i % len(d_colors)]
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
            marker = markers[i % len(markers)]
            
            runs.append({
                'name': name,
                'file': filename,
                'filepath': log_dir / filename,
                'train_color': color,
                'val_color': color,
                'marker': marker,
                'is_final': False,
                'linewidth': 2.5,
            })




# --- 4. Special Manual Entries ---

# Origin Gemma
if (log_dir / 'origin_gemma.out').exists():
    runs.append({
        'name': 'Origin Gemma',
        'file': 'origin_gemma.out',
        'filepath': log_dir / 'origin_gemma.out',
        'train_color': 'tab:orange',
        'val_color': 'tab:orange',
        'marker': 'D',
        'is_final': False,
        'linewidth': 3.0,
    })

# Hybrid Norm Mixed entries (Manual addition)
hybrid_configs = [
    ('0.012', 'cyan', 's'),
    ('0.013', 'blue', 'v'),
    ('0.015', 'navy', '^')
]

for lr, color, marker in hybrid_configs:
    fname = f'mixed_hybrid_norm_muonlr{lr}_wd1.44.out'
    if (log_dir / fname).exists():
        runs.append({
            'name': f'Hybrid Norm {lr}',
            'file': fname,
            'filepath': log_dir / fname,
            'train_color': color,
            'val_color': color,
            'marker': marker,
            'is_final': False,
            'linewidth': 3.0,
        })

# --- Parsing and Plotting Functions (Unchanged) ---

def parse_params_from_filename(filename):
    """Parse hyperparameters from filename like 'final_muonlr0.013_wd1.44.out'."""
    match = re.search(r'final_muonlr([\d.]+?)_wd([\d.]+?)(?:\.out|$)', filename)
    if match:
        return float(match.group(1).rstrip('.')), float(match.group(2).rstrip('.'))
    return None, None

def get_label_from_filename(filename, val_loss_final):
    muon_lr, weight_decay = parse_params_from_filename(filename)
    if muon_lr is not None and weight_decay is not None:
        return f"final_lr{muon_lr:.3f}_wd{weight_decay:.2f} (val: {val_loss_final:.4f})"
    name = filename.replace('.out', '').replace('.txt', '')
    # Simplify long names for legend
    name = name.replace('mixed_', '').replace('muonlr', 'lr').replace('velocitymomentum', 'vm')
    return f"{name} ({val_loss_final:.4f})"

def extract_training_data(filepath):
    train_steps, train_loss, val_steps, val_loss = [], [], [], []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'step:' in line:
                    tm = re.search(r'step:(\d+)/\d+ train_loss:([\d.]+)', line)
                    if tm:
                        train_steps.append(int(tm.group(1)))
                        train_loss.append(float(tm.group(2)))
                    vm = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
                    if vm:
                        val_steps.append(int(vm.group(1)))
                        val_loss.append(float(vm.group(2)))
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
    return train_steps, train_loss, val_steps, val_loss

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 24))

final_losses = []

print("\nExtracting data from runs...")
# Filter runs to only plot those that exist
valid_runs = []
for run in runs:
    if run['filepath'].exists():
        valid_runs.append(run)
    else:
        # Only warn if it's not a glob pattern result (which we know exists)
        if 'record' in run['name']:
             print(f"Warning: Baseline file not found: {run['filepath']}")

for run in valid_runs:
    train_steps, train_loss, val_steps, val_loss = extract_training_data(run['filepath'])

    if not val_steps:
        print(f"  {run['name']}: No validation data yet")
        continue

    linewidth = run.get('linewidth', 2.5)
    alpha = 0.8
    markersize = 6
    
    if run['is_final']:
        label = get_label_from_filename(run['file'], val_loss[-1])
    else:
        label = f"{run['name']} ({val_loss[-1]:.4f})"

    # Plot on both axes
    for ax in [ax1, ax2]:
        ax.plot(val_steps, val_loss, color=run['val_color'], linewidth=linewidth, alpha=alpha,
                marker=run['marker'], markersize=markersize, label=label, markevery=max(1, len(val_steps)//10))

    if run['is_final']:
        final_losses.append(val_loss[-1])

    print(f"  {run['name']}: Final Val: {val_loss[-1]:.6f}")

# Style axes
for ax in [ax1, ax2]:
    ax.set_xlabel('Step', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9, ncol=2)

ax1.set_title('Validation Loss Comparison', fontsize=18)
ax1.set_ylim([2.9, 3.2]) # Zoomed view

ax2.set_title('Validation Loss (Wide View)', fontsize=18)
ax2.set_ylim([2.9, 4.0]) # Wide view

plt.tight_layout()
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, format='png', bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

