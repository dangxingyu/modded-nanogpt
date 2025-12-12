import re
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import numpy as np

# File paths
log_dir = Path(__file__).parent / 'logs'
output_path = Path(__file__).parent / 'figure' / 'ema_sweep_comparison.png'

# Define sweep parameters for EMA runs
MUON_LRS = [0.010, 0.011, 0.012, 0.013]
WDS = [1.2, 1.3, 1.4, 1.5]
EMA_VALS = [0.85]

# Generate all possible sweep file names
sweep_files = []
for muon_lr in MUON_LRS:
    for wd in WDS:
        for ema in EMA_VALS:
            filename = f'ema_muonlr{muon_lr:.3f}_wd{wd}_ema{ema}.out'
            sweep_files.append(filename)

# Generate colors for sweep runs using a colormap
num_sweep = len(sweep_files)
colors = cm.viridis(np.linspace(0, 1, num_sweep))

runs = []
for i, filename in enumerate(sweep_files):
    runs.append({
        'name': f'EMA Run {i+1}',
        'file': filename,
        'train_color': colors[i],
        'val_color': colors[i],
        'marker': 'o',
        'is_sweep': True
    })

# Add Record 8
runs.append({
    'name': 'Record 8',
    'file': '../records/track_2_medium/2025-06-15_OptimizationLeaderboard/075_640429f2-e726-4e83-aa27-684626239ffc.txt',
    'train_color': 'black',
    'val_color': 'black',
    'marker': 'H',
    'is_sweep': False,
    'linewidth': 3.0,
})

def parse_params_from_filename(filename):
    """Parse hyperparameters from filename like 'ema_muonlr0.010_wd1.2_ema0.85.out'."""
    # Match pattern: ema_muonlr{float}_wd{float}_ema{float}.out
    match = re.search(r'ema_muonlr([\d.]+)_wd([\d.]+)_ema([\d.]+)\.out', filename)
    if match:
        muon_lr = float(match.group(1))
        wd = float(match.group(2))
        ema = float(match.group(3))
        return muon_lr, wd, ema
    
    return None, None, None

def get_label_from_filename(filename, val_loss_final):
    """Generate label from filename, handling EMA sweep format."""
    muon_lr, wd, ema = parse_params_from_filename(filename)
    
    # Handle EMA sweep format
    if muon_lr is not None and wd is not None and ema is not None:
        return f"lr{muon_lr:.3f}_wd{wd}_ema{ema} (val: {val_loss_final:.4f})"
    
    # Fallback: use filename without extension
    name = filename.replace('.out', '').replace('.txt', '')
    return f"{name} (val: {val_loss_final:.4f})"

def extract_training_data(filepath):
    """Extract training and validation loss from log file."""
    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []
    val_ema_steps = []
    val_loss_ema = []

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

                    # Parse val loss ema
                    val_ema_match = re.search(r'step:(\d+)/\d+ val_loss_ema:([\d.]+)', line)
                    if val_ema_match:
                        step = int(val_ema_match.group(1))
                        loss = float(val_ema_match.group(2))
                        val_ema_steps.append(step)
                        val_loss_ema.append(loss)
    except FileNotFoundError:
        pass

    return train_steps, train_loss, val_steps, val_loss, val_ema_steps, val_loss_ema

# Create the plot
fig, ax = plt.subplots(figsize=(20, 12))

# Collect all sweep results for statistics
sweep_final_losses = []
sweep_files_info = []  # Store (filename, val_loss) tuples

# Plot all runs
print("Extracting data from runs...")
for run in runs:
    filepath = log_dir / run['file']

    train_steps, train_loss, val_steps, val_loss, val_ema_steps, val_loss_ema = extract_training_data(filepath)

    if not train_steps and not val_steps:
        print(f"  {run['name']}: No data yet (still compiling or not started)")
        continue

    linewidth = run.get('linewidth', 3.5 if not run['is_sweep'] else 2.5)
    alpha = 0.7 if run['is_sweep'] else 1.0
    markersize = 5 if run['is_sweep'] else 8

    # Only plot validation loss (no training loss)
    if val_steps:
        # Generate label from filename
        if run['is_sweep']:
            label = get_label_from_filename(run['file'], val_loss[-1])
        else:
            # For non-sweep runs, just use the run name
            label = f"{run['name']} (val: {val_loss[-1]:.4f})"

        ax.plot(val_steps, val_loss, color=run['val_color'], linewidth=linewidth, alpha=alpha,
                marker=run['marker'], markersize=markersize, label=label)
        
        if val_loss_ema:
            ax.plot(val_ema_steps, val_loss_ema, color=run['val_color'], linewidth=linewidth, alpha=alpha,
                    linestyle='--', marker=run['marker'], markersize=markersize, label=f"{run['name']} EMA (val: {val_loss_ema[-1]:.4f})")

        # Collect sweep statistics with file info
        if run['is_sweep'] and val_loss:
            final_loss = val_loss[-1]
            sweep_final_losses.append(final_loss)
            sweep_files_info.append((run['file'], final_loss))

        # Print stats
        print(f"  {run['name']}:")
        if train_loss:
            print(f"    Train: {len(train_steps)} points, Final: {train_loss[-1]:.4f}")
        if val_loss:
            print(f"    Val: {len(val_steps)} points, Final: {val_loss[-1]:.6f}")

# Print sweep statistics
if sweep_final_losses:
    print("\nSweep Statistics:")
    print(f"  Number of runs: {len(sweep_final_losses)}")
    print(f"  Best val loss: {min(sweep_final_losses):.6f}")
    print(f"  Worst val loss: {max(sweep_final_losses):.6f}")
    print(f"  Mean val loss: {np.mean(sweep_final_losses):.6f}")
    print(f"  Std val loss: {np.std(sweep_final_losses):.6f}")
    print(f"  Median val loss: {np.median(sweep_final_losses):.6f}")

ax.set_xlabel('Step', fontsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title('EMA Sweep Results vs Record 8',
             fontsize=20, fontweight='bold')

# Show legend with all runs
ax.legend(fontsize=11, loc='lower left', framealpha=0.95, ncol=2)

ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=16)

# Zoom to show details
ax.set_ylim([2.9, 3.1])

plt.tight_layout()

# Save figure
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, format='png', bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

# Compare with Record 8
record_8_path = log_dir / '../records/track_2_medium/2025-06-15_OptimizationLeaderboard/075_640429f2-e726-4e83-aa27-684626239ffc.txt'
_, _, _, record_8_val, _, _ = extract_training_data(record_8_path)
if record_8_val and sweep_final_losses:
    record_8_final = record_8_val[-1]
    best_sweep = min(sweep_final_losses)
    improvement = record_8_final - best_sweep
    print(f"\nComparison:")
    print(f"  Record 8 final val loss: {record_8_final:.6f}")
    print(f"  Best sweep final val loss: {best_sweep:.6f}")
    print(f"  Improvement: {improvement:+.6f}")

