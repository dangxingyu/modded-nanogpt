import re
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import numpy as np

# File paths
log_dir = Path(__file__).parent / 'logs'
output_path = Path(__file__).parent / 'figure' / 'training_comparison_sweep.png'

# Define sweep runs - using output files with parameters in filename
sweep_files = [
    'zslr0.03_mlr0.004_cd0.7.out',
    'zslr0.03_mlr0.004_cd0.8.out',
    'zslr0.03_mlr0.006_cd0.7.out',
    # 'zslr0.03_mlr0.006_cd0.8.out',
    # 'zslr0.03_mlr0.012_cd0.7.out',
    # 'zslr0.03_mlr0.012_cd0.8.out',
    # 'zslr0.03_mlr0.018_cd0.7.out',
    # 'zslr0.03_mlr0.018_cd0.8.out',
    # 'zslr0.06_mlr0.004_cd0.7.out',
    # 'zslr0.06_mlr0.004_cd0.8.out',
    # 'zslr0.06_mlr0.006_cd0.7.out',
    # 'zslr0.06_mlr0.006_cd0.8.out',
    # 'zslr0.06_mlr0.012_cd0.7.out',
    'renorm-zslr0.03_mlr0.004_cd0.7.out',
    'renorm-zslr0.03_mlr0.006_cd0.7.out',
    'renorm-zslr0.03_mlr0.012_cd0.7.out',
    'renorm-zslr0.03_mlr0.018_cd0.7.out',
    'weird_lr_unified.out',
    'seperate_lr.out',
    'baseline_cooldown0.6.out',
    'baseline_cooldown0.4.out',
    'baseline_cooldown0.6-renorm.out',
    'baseline_muonlr0.015_cooldown0.8_cosine.out'
]

# Generate colors for sweep runs using a colormap
num_sweep = len(sweep_files)
colors = cm.viridis(np.linspace(0, 1, num_sweep))

runs = []
for i, filename in enumerate(sweep_files):
    runs.append({
        'name': f'Sweep Run {i+1}',
        'file': filename,
        'train_color': colors[i],
        'val_color': colors[i],
        'marker': 'o',
        'is_sweep': True
    })

# Add record_7
runs.append({
    'name': 'Record 7',
    'file': 'record_7.txt',
    'train_color': 'black',
    'val_color': 'black',
    'marker': 'H',
    'is_sweep': False,
    'linewidth': 3.0,
})

def parse_params_from_filename(filename):
    """Parse hyperparameters from filename like 'zslr0.06_mlr0.012_cd0.7.out'."""
    # Match pattern: zslr{float}_mlr{float}_cd{float} (don't include .out extension)
    match = re.search(r'(?:renorm-)?zslr(\d+\.?\d*)_mlr(\d+\.?\d*)_cd(\d+\.?\d*)', filename)
    if match:
        zero_lr = float(match.group(1))
        muon_lr = float(match.group(2))
        cooldown = float(match.group(3))
        return zero_lr, muon_lr, cooldown, 'renorm' in filename
    
    # For special filenames like 'weird_lr_unified.out' or 'seperate_lr.out'
    # Return a special marker
    special_names = ['weird_lr_unified.out', 'seperate_lr.out', 'baseline_cooldown0.6.out', 'baseline_cooldown0.4.out', 'baseline_cooldown0.6-renorm.out', 'baseline_muonlr0.015_cooldown0.8_cosine.out']
    if filename in special_names:
        return 'special', None, None, False
    
    return None, None, None, False

def get_label_from_filename(filename, val_loss_final):
    """Generate label from filename, handling various formats."""
    zero_lr, muon_lr, cooldown, is_renorm = parse_params_from_filename(filename)
    
    # Handle special filenames
    if zero_lr == 'special':
        # Use filename without extension as label
        name = filename.replace('.out', '').replace('.txt', '')
        return f"{name} (val: {val_loss_final:.4f})"
    
    # Handle standard parameter format
    if zero_lr is not None and muon_lr is not None and cooldown is not None:
        prefix = "renorm-" if is_renorm else ""
        return f"{prefix}z{zero_lr:.2f}_m{muon_lr:.3f}_c{cooldown:.1f} (val: {val_loss_final:.4f})"
    
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
        pass

    return train_steps, train_loss, val_steps, val_loss

# Create the plot
fig, ax = plt.subplots(figsize=(20, 12))

# Collect all sweep results for statistics
sweep_final_losses = []
sweep_files_info = []  # Store (filename, val_loss, is_special) tuples

# Plot all runs
print("Extracting data from runs...")
for run in runs:
    filepath = log_dir / run['file']

    train_steps, train_loss, val_steps, val_loss = extract_training_data(filepath)

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

        # Collect sweep statistics with file info
        if run['is_sweep'] and val_loss:
            final_loss = val_loss[-1]
            sweep_final_losses.append(final_loss)
            # Check if it's a special name file
            zero_lr, _, _, _ = parse_params_from_filename(run['file'])
            is_special = (zero_lr == 'special')
            sweep_files_info.append((run['file'], final_loss, is_special))

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
    
    # Separate statistics for standard and special files
    standard_losses = [loss for _, loss, is_special in sweep_files_info if not is_special]
    special_losses = [loss for _, loss, is_special in sweep_files_info if is_special]
    
    if standard_losses:
        print("\n  Standard Parameter Files:")
        print(f"    Number: {len(standard_losses)}")
        print(f"    Best: {min(standard_losses):.6f}")
        print(f"    Worst: {max(standard_losses):.6f}")
        print(f"    Mean: {np.mean(standard_losses):.6f}")
    
    if special_losses:
        print("\n  Special Name Files:")
        print(f"    Number: {len(special_losses)}")
        print(f"    Best: {min(special_losses):.6f}")
        print(f"    Worst: {max(special_losses):.6f}")
        print(f"    Mean: {np.mean(special_losses):.6f}")
        print("    Files:")
        for filename, loss, _ in sweep_files_info:
            if loss in special_losses:
                print(f"      - {filename}: {loss:.6f}")

ax.set_xlabel('Step', fontsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title('Hyperparameter Sweep Results vs Record 7',
             fontsize=20, fontweight='bold')

# Show legend with all runs
ax.legend(fontsize=11, loc='upper right', framealpha=0.95, ncol=2)

ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=16)

# Zoom to show details
ax.set_ylim([2.5, 4.5])

plt.tight_layout()

# Save figure
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, format='png', bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

# Compare with Record 7
record_7_path = log_dir / 'record_7.txt'
_, _, _, record_7_val = extract_training_data(record_7_path)
if record_7_val and sweep_final_losses:
    record_7_final = record_7_val[-1]
    best_sweep = min(sweep_final_losses)
    improvement = record_7_final - best_sweep
    print(f"\nComparison:")
    print(f"  Record 7 final val loss: {record_7_final:.6f}")
    print(f"  Best sweep final val loss: {best_sweep:.6f}")
    print(f"  Improvement: {improvement:+.6f}")
