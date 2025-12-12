import re
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
log_dir = Path(__file__).parent / 'logs'
prev_record_path = log_dir / 'prev_record.txt'
output_path = Path(__file__).parent / 'figure' / 'training_comparison_1000steps.png'

# Define all runs to plot
runs = [
    # Removed: No Zero Init (blue line)
    {
        'name': 'Hyperball (Muon lr=0.01)',
        'file': '000_7c35ee2b-fd27-4110-b782-76c206fbb56b.txt',
        'train_color': 'g',
        'val_color': 'g',
        'marker': '^',
    },
    {
        'name': 'Baseline (Muon lr=0.01) -new',
        'file': '000_9521e316-d84b-44ad-9dab-6fdc0dae203b.txt',
        'train_color': 'blue',
        'val_color': 'blue',
        'marker': 'o',
    },
    {
        'name': 'Gemma RMSNorm (Muon lr=0.005)',
        'file': '000_be766d78-1518-45cf-800c-adfcb2ca5506.txt',
        'train_color': 'brown',
        'val_color': 'brown',
        'marker': 'p',
    },
    {
        'name': 'Gemma RMSNorm (Muon lr=0.01)',
        'file': '000_30dca714-1e05-44b2-ad90-f0aa7c646e53.txt',
        'train_color': 'purple',
        'val_color': 'purple',
        'marker': 's',
    },
    {
        'name': 'Gemma RMSNorm (Muon lr=0.02)',
        'file': '000_8b236d45-72b7-4184-a920-1a959e529b2e.txt',
        'train_color': 'orange',
        'val_color': 'orange',
        'marker': 'd',
    },
    {
        'name': 'Gemma RMSNorm (Muon lr=0.04)',
        'file': '000_ab204632-34c7-4fc7-8d96-47f6d90e5f2c.txt',
        'train_color': 'cyan',
        'val_color': 'cyan',
        'marker': 'v',
    },
    {
        'name': 'Hyperball (Muon lr=0.002) -rescheduled',
        'file': '000_f91c89a9-f1f1-4290-b9a0-5461746f2046.txt',
        'train_color': 'darkgreen',
        'val_color': 'darkgreen',
        'marker': 'D',
    },
    {
        'name': 'Hyperball (Muon lr=0.005) -rescheduled',
        'file': '000_37b220b1-f326-4294-bcbd-27659cea6ec0.txt',
        'train_color': 'gold',
        'val_color': 'gold',
        'marker': 'h',
    },
]

MAX_STEPS = 1000  # Only plot first 1000 steps

def extract_training_data(filepath, max_steps=None):
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
                        if max_steps is None or step <= max_steps:
                            loss = float(train_match.group(2))
                            train_steps.append(step)
                            train_loss.append(loss)

                    # Parse val loss
                    val_match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
                    if val_match:
                        step = int(val_match.group(1))
                        if max_steps is None or step <= max_steps:
                            loss = float(val_match.group(2))
                            val_steps.append(step)
                            val_loss.append(loss)
    except FileNotFoundError:
        pass

    return train_steps, train_loss, val_steps, val_loss

def extract_val_only(filepath, max_steps=None):
    """Extract only validation loss from log file."""
    val_steps = []
    val_loss = []

    with open(filepath, 'r') as f:
        for line in f:
            if 'val_loss:' in line and 'step:' in line:
                match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    if max_steps is None or step <= max_steps:
                        loss = float(match.group(2))
                        val_steps.append(step)
                        val_loss.append(loss)

    return val_steps, val_loss

# Create the plot
fig, ax = plt.subplots(figsize=(18, 10))

# Plot all runs
print("Extracting data from runs (first 1000 steps)...")
for run in runs:
    filepath = log_dir / run['file']

    train_steps, train_loss, val_steps, val_loss = extract_training_data(filepath, max_steps=MAX_STEPS)

    if not train_steps and not val_steps:
        print(f"  {run['name']}: No data yet (still compiling or not started)")
        continue

    if train_steps:
        # Plot train loss (lighter, thinner)
        ax.plot(train_steps, train_loss, color=run['train_color'], alpha=0.2, linewidth=1,
                label=f"{run['name']} - Train")

    if val_steps:
        # Plot val loss (darker, thicker, with markers)
        ax.plot(val_steps, val_loss, color=run['val_color'], linewidth=2.5,
                marker=run['marker'], markersize=5, label=f"{run['name']} - Val")

        # Print stats
        print(f"  {run['name']}:")
        if train_loss:
            print(f"    Train: {len(train_steps)} points, Final (at step {train_steps[-1]}): {train_loss[-1]:.4f}")
        if val_loss:
            print(f"    Val: {len(val_steps)} points, Final (at step {val_steps[-1]}): {val_loss[-1]:.6f}")

# Plot previous record
print("\nExtracting data from previous record (first 1000 steps)...")
prev_val_steps, prev_val_loss = extract_val_only(prev_record_path, max_steps=MAX_STEPS)
ax.plot(prev_val_steps, prev_val_loss, 'r-', linewidth=2.5, marker='*', markersize=7,
        label='Previous Record - Val', zorder=10)
if prev_val_loss:
    print(f"  Val: {len(prev_val_steps)} points, Final (at step {prev_val_steps[-1]}): {prev_val_loss[-1]:.6f}")

ax.set_xlabel('Step', fontsize=16)
ax.set_ylabel('Loss', fontsize=16)
ax.set_title('Training Comparison (First 1000 Steps): Gemma RMSNorm Runs vs Previous Record', fontsize=18, fontweight='bold')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=14)

# Set x-axis limit to 1000 steps
ax.set_xlim([0, MAX_STEPS])

# Auto-adjust y-axis based on data, but don't set a hard limit
# This allows the y-axis to adapt to the data range

plt.tight_layout()

# Save figure
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

# Print improvements at the closest step to 1000
if prev_val_loss:
    # Find the validation loss closest to step 1000 for previous record
    prev_idx = min(range(len(prev_val_steps)), key=lambda i: abs(prev_val_steps[i] - MAX_STEPS), default=None)
    if prev_idx is not None:
        prev_final = prev_val_loss[prev_idx]
        prev_step = prev_val_steps[prev_idx]
        print(f"\nComparison to Previous Record at step {prev_step} (val loss: {prev_final:.6f}):")
        for run in runs:
            filepath = log_dir / run['file']
            _, _, val_steps, val_loss = extract_training_data(filepath, max_steps=MAX_STEPS)
            if val_loss and val_steps:
                # Find validation loss closest to the same step
                idx = min(range(len(val_steps)), key=lambda i: abs(val_steps[i] - prev_step))
                improvement = prev_final - val_loss[idx]
                status = "✓" if improvement > 0 else "✗"
                print(f"  {status} {run['name']} at step {val_steps[idx]}: {val_loss[idx]:.6f} (improvement: {improvement:+.6f})")
