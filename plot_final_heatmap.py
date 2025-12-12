import re
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import numpy as np
import glob
import pandas as pd
import seaborn as sns

# File paths
log_dir = Path(__file__).parent / 'logs'
output_path = Path(__file__).parent / 'figure' / 'training_comparison_final_heatmap.png'
heatmap_output_path = Path(__file__).parent / 'figure' / 'final_heatmap.png'
record_file = Path(__file__).parent / 'records' / 'track_2_medium' / '2025-06-15_OptimizationLeaderboard' / '075_640429f2-e726-4e83-aa27-684626239ffc.txt'

# Find all final_*.out files
final_files = sorted(glob.glob(str(log_dir / 'final_*.out')))
final_files = [Path(f).name for f in final_files]  # Get just the filename

print(f"Found {len(final_files)} final files:")
for f in final_files:
    print(f"  - {f}")

# Generate colors for final runs using a colormap
num_final = len(final_files)
colors = cm.viridis(np.linspace(0, 1, num_final)) if num_final > 0 else []

runs = []

# Add all final_*.out files
for i, filename in enumerate(final_files):
    runs.append({
        'name': f'Final {i+1}',
        'file': filename,
        'filepath': log_dir / filename,
        'train_color': colors[i] if i < len(colors) else 'blue',
        'val_color': colors[i] if i < len(colors) else 'blue',
        'marker': 'o',
        'is_final': True,
        'linewidth': 2.5,
    })

# Add record-8
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
# 遍历log_dir下所有的 mixed_muonlr{}_wd{}.out 文件，并自动加入 runs

mixed_files = sorted(glob.glob(str(log_dir / 'mixed_muonlr*_wd*.out')))
mixed_w_o_gemma_files = sorted(glob.glob(str(log_dir / 'mixed_w_o_gemma_muonlr*_wd*.out')))
last_mixed = sorted(glob.glob(str(log_dir / 'last_hybrid_norm_muonlr0.0*_wd*.out')))
mixed_files = mixed_files + mixed_w_o_gemma_files + last_mixed
mixed_colors = cm.Paired(np.linspace(0, 1, max(len(mixed_files), 1)))  # 用Paired色系区分

for i, path in enumerate(mixed_files):
    filename = Path(path).name
    match = re.search(r'muonlr([0-9.]+)_wd([0-9.]+)', filename)
    if match:
        muonlr = match.group(1)
        wd = match.group(2)
        name = f"Mixed {muonlr} w/o Gemma" if 'w_o_gemma' in path else f"Last Hybrid Norm {muonlr} wd{wd}" if 'last_hybrid_norm' in path else f"Mixed {muonlr} wd{wd}"
        color = mixed_colors[i % len(mixed_colors)]
        marker = ['v', 's', '^', 'd', '>', '<', '*', 'P', 'X', '8'][i % 10]  # 常用marker轮转
        runs.append({
            'name': name,
            'file': filename,
            'filepath': log_dir / filename,
            'train_color': color,
            'val_color': color,
            'marker': marker,
            'is_final': False,
            'linewidth': 3.0,
        })

# Append origin_gemma.out from logs
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
    

runs.append({
    'name': 'Hybrid Norm Mixed 0.012',
    'file': 'mixed_hybrid_norm_muonlr0.012_wd1.44.out',
    'filepath': log_dir / 'mixed_hybrid_norm_muonlr0.012_wd1.44.out',
    'train_color': 'cyan',
    'val_color': 'cyan',
    'marker': 's',
    'is_final': False,
    'linewidth': 3.0,
})

runs.append({
    'name': 'Hybrid Norm Mixed 0.013',
    'file': 'mixed_hybrid_norm_muonlr0.013_wd1.44.out',
    'filepath': log_dir / 'mixed_hybrid_norm_muonlr0.013_wd1.44.out',
    'train_color': 'blue',
    'val_color': 'blue',
    'marker': 'v',
    'is_final': False,
    'linewidth': 3.0,
})
runs.append({
    'name': 'Hybrid Norm Mixed 0.015',
    'file': 'mixed_hybrid_norm_muonlr0.015_wd1.44.out',
    'filepath': log_dir / 'mixed_hybrid_norm_muonlr0.015_wd1.44.out',
    'train_color': 'blue',
    'val_color': 'blue',
    'marker': 'v',
    'is_final': False,
    'linewidth': 3.0,
})

def parse_params_from_filename(filename):
    """Parse hyperparameters from filename like 'final_muonlr0.013_wd1.44.out'."""
    # Match pattern: final_muonlr{float}_wd{float}.out
    # Use non-greedy matching to avoid capturing trailing dot from .out
    match = re.search(r'final_muonlr([\d.]+?)_wd([\d.]+?)(?:\.out|$)', filename)
    if match:
        muon_lr_str = match.group(1).rstrip('.')
        weight_decay_str = match.group(2).rstrip('.')
        muon_lr = float(muon_lr_str)
        weight_decay = float(weight_decay_str)
        return muon_lr, weight_decay
    
    return None, None

def get_label_from_filename(filename, val_loss_final):
    """Generate label from filename, handling final format."""
    muon_lr, weight_decay = parse_params_from_filename(filename)
    
    # Handle final format
    if muon_lr is not None and weight_decay is not None:
        return f"final_lr{muon_lr:.3f}_wd{weight_decay:.2f} (val: {val_loss_final:.4f})"
    
    # Fallback: use filename without extension
    name = filename.replace('.out', '').replace('.txt', '')
    return f"{name} (val: {val_loss_final:.4f})"

def extract_training_data(filepath):
    """Extract training and validation loss from log file."""
    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'step:' in line:
                    # Parse train loss
                    train_match = re.search(r'step:(\d+)/\d+ train_loss:([\d.]+)', line)
                    if train_match:
                        step = int(train_match.group(1))
                        loss = float(train_match.group(2))
                        train_steps.append(step)
                        train_loss.append(loss)

                    # Parse val loss
                    val_match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
                    if val_match:
                        step = int(val_match.group(1))
                        loss = float(val_match.group(2))
                        val_steps.append(step)
                        val_loss.append(loss)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")

    return train_steps, train_loss, val_steps, val_loss

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 24))

# Collect all final results for statistics
final_losses = []
final_files_info = []  # Store (filename, val_loss) tuples

# Separate final runs and record-8 for proper legend ordering
final_runs = [r for r in runs if r['is_final']]
record_runs = [r for r in runs if not r['is_final']]

# Heatmap data collection
heatmap_data = []

# Plot all runs: first final runs, then record-8 (so it appears last in legend)
print("\nExtracting data from runs...")
for run in final_runs + record_runs:
    filepath = run['filepath']

    train_steps, train_loss, val_steps, val_loss = extract_training_data(filepath)

    if not train_steps and not val_steps:
        print(f"  {run['name']}: No data yet (still compiling or not started)")
        continue

    # Collect data for heatmap from last_hybrid_norm runs
    if 'last_hybrid_norm' in run['file']:
        match = re.search(r'muonlr([0-9.]+)_wd([0-9.]+)', run['file'])
        if match and val_loss:
            muon_lr = float(match.group(1).rstrip('.'))
            wd = float(match.group(2).rstrip('.'))
            final_val_loss = val_loss[-1]
            heatmap_data.append({'muon_lr': muon_lr, 'wd': wd, 'val_loss': final_val_loss})

    linewidth = run.get('linewidth', 3.5 if not run['is_final'] else 2.5)
    alpha = 0.7 if run['is_final'] else 1.0
    markersize = 5 if run['is_final'] else 8

    # Only plot validation loss (no training loss)
    if val_steps:
        # Generate label from filename
        if run['is_final']:
            label = get_label_from_filename(run['file'], val_loss[-1])
        else:
            # For non-final runs, just use the run name
            label = f"{run['name']} (val: {val_loss[-1]:.4f})"

        ax1.plot(val_steps, val_loss, color=run['val_color'], linewidth=linewidth, alpha=alpha,
                marker=run['marker'], markersize=markersize, label=label)
        ax2.plot(val_steps, val_loss, color=run['val_color'], linewidth=linewidth, alpha=alpha,
                marker=run['marker'], markersize=markersize, label=label)

        # Collect final statistics with file info
        if run['is_final'] and val_loss:
            final_loss = val_loss[-1]
            final_losses.append(final_loss)
            final_files_info.append((run['file'], final_loss))

        # Print stats
        print(f"  {run['name']}:")
        if train_loss:
            print(f"    Train: {len(train_steps)} points, Final: {train_loss[-1]:.4f}")
        if val_loss:
            print(f"    Val: {len(val_steps)} points, Final: {val_loss[-1]:.6f}")

# Print final statistics
if final_losses:
    print("\nFinal Statistics:")
    print(f"  Number of runs: {len(final_losses)}")
    print(f"  Best val loss: {min(final_losses):.6f}")
    print(f"  Worst val loss: {max(final_losses):.6f}")
    print(f"  Mean val loss: {np.mean(final_losses):.6f}")
    print(f"  Std val loss: {np.std(final_losses):.6f}")
    print(f"  Median val loss: {np.median(final_losses):.6f}")

for ax in [ax1, ax2]:
    ax.set_xlabel('Step', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=16)
    # Show legend with all runs
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95, ncol=2)

ax1.set_title('Final Sweep Results vs record-8',
             fontsize=20, fontweight='bold')
ax2.set_title('Final Sweep Results vs record-8 (Wide Y-Axis)',
             fontsize=20, fontweight='bold')

# Set y-axis range
ax1.set_ylim([2.9, 3.2])
ax2.set_ylim([2.9, 3.7])

plt.tight_layout()

# Save figure
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, format='png', bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

# Generate Heatmap
if heatmap_data:
    print("\nGenerating Heatmap for Last Hybrid Norm Runs...")
    df = pd.DataFrame(heatmap_data)
    
    # Pivot the data for heatmap
    pivot_table = df.pivot(index='muon_lr', columns='wd', values='val_loss')
    
    # Sort index and columns
    pivot_table = pivot_table.sort_index(ascending=True)
    pivot_table = pivot_table.sort_index(axis=1, ascending=True)
    
    plt.figure(figsize=(10, 8))
    # Set vmin and vmax to filter outliers visually, colors outside range will be clamped or handled by cmap
    # To make outliers black, we can mask them or use a custom cmap, but seaborn heatmap doesn't support 'bad' color easily for values outside vmin/vmax with simple args.
    # Instead, we can use vmin=2.92, vmax=2.93 and let seaborn clip the colormap.
    # To explicitly make outliers black, we can modify the dataframe or mask.
    
    # Simple approach: set vmin/vmax. Values outside will be colored with the extremes of the colormap.
    # To strictly make them black, we need a custom colormap.
    
    cmap = cm.get_cmap("viridis_r").copy()
    cmap.set_over('black')
    cmap.set_under('black')
    
    # We need to mask values outside the range for set_over/set_under to work if we use matplotlib directly,
    # but seaborn heatmap is a wrapper.
    # Let's try passing vmin/vmax directly.
    
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r", vmin=2.92, vmax=2.93, cbar_kws={'label': 'Validation Loss', 'extend': 'both'})
    
    # Note: Seaborn's heatmap with vmin/vmax will clamp colors to the extremes, not necessarily black for 'outliers'.
    # If 'black' is strictly required for > 2.93 or < 2.92 (outliers), we can use a mask.
    # But usually 'outlier' in loss context means 'loss exploded', so > 2.93.
    # Let's assume we want to focus on the 2.92-2.93 range.
    plt.title('Validation Loss Heatmap: Muon LR vs Weight Decay (Last Hybrid Norm)', fontsize=16, fontweight='bold')
    plt.xlabel('Weight Decay (wd_mul)', fontsize=14)
    plt.ylabel('Muon Learning Rate', fontsize=14)
    plt.tight_layout()
    
    heatmap_output_path.parent.mkdir(exist_ok=True)
    plt.savefig(heatmap_output_path, format='png', bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_output_path}")
else:
    print("\nNo data found for heatmap (Last Hybrid Norm runs).")

# Compare with record-8
record_8_path = record_file
_, _, _, record_8_val = extract_training_data(record_8_path)
if record_8_val and final_losses:
    record_8_final = record_8_val[-1]
    best_final = min(final_losses)
    improvement = record_8_final - best_final
    print(f"\nComparison:")
    print(f"  record-8 final val loss: {record_8_final:.6f}")
    print(f"  Best final sweep final val loss: {best_final:.6f}")
    print(f"  Improvement: {improvement:+.6f}")

