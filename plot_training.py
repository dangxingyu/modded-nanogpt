import re
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
log_dir = Path(__file__).parent / 'logs'
prev_record_path = log_dir / 'prev_record.txt'
# output_path = Path(__file__).parent / 'figure' / 'training_comparison.pdf'
output_path = '/scratch/gpfs/ARORA/xd7812/speedrun/modded-nanogpt/figure/training_comparison.pdf'

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
    # {
    #     'name': 'Gemma RMSNorm (Muon lr=0.04)',
    #     'file': '000_ab204632-34c7-4fc7-8d96-47f6d90e5f2c.txt',
    #     'train_color': '#1f77b4',  # changed color to matplotlib default blue
    #     'val_color': '#1f77b4',
    #     'marker': 'v',
    # },
    # New runs with norm logging
    # {
    #     'name': 'Origin (Muon lr=0.025) -norm-log',
    #     'file': '000_91a27ca8-3c5a-4c0f-923f-708894f1951d.txt',
    #     'train_color': 'lime',
    #     'val_color': 'lime',
    #     'marker': 'P',
    # },
    # {
    #     'name': 'Gemma RMSNorm (Muon lr=0.005) -norm-log',
    #     'file': '000_5c54e40d-de02-4292-a0e3-1e79ca9ff302.txt',
    #     'train_color': 'pink',
    #     'val_color': 'pink',
    #     'marker': 'X',
    # },
    # {
    #     'name': 'Gemma RMSNorm (Muon lr=0.01) -norm-log',
    #     'file': '000_d3659f31-4462-4cba-93cd-612dc081ae78.txt',
    #     'train_color': 'gold',
    #     'val_color': 'gold',
    #     'marker': 'h',
    # },
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
    {
        'name': 'Hyperball (Muon lr=0.01) -rescheduled',
        'file': '000_6ec0d7dd-d416-4bf4-8272-855f54d3f7c3.txt',
        'train_color': 'pink',
        'val_color': 'pink',
        'marker': 'H',
    },
    {
        'name': 'Hyperball (Muon lr=0.01) -switch',
        'file': '000_b2a522c8-ddf2-40ef-b08c-22b083e4d26f.txt',
        'train_color': 'cyan',
        'val_color': 'cyan',
        'marker': 'H',
    },
    # {
    #     'name': 'Hyperball (Muon lr=0.01) -numerical',
    #     'file': '000_9eae30b2-53f4-4ffb-a93d-f986bfd6c35e.txt',
    #     'train_color': 'red',
    #     'val_color': 'red',
    #     'marker': 'H',
    # },
    {
        'name': 'Switch 2 (Muon lr=0.01)',
        'file': '000_8b7e13b7-601b-470e-9553-4c0fed77fbe4.txt',
        'train_color': 'magenta',
        'val_color': 'magenta',
        'marker': 'H',
    },
    {
        'name': 'Switch 3 (Muon lr=0.01)',
        'file': '000_23fb6270-1ddf-47cc-972e-5772d332beea.txt',
        'train_color': 'yellow',
        'val_color': 'yellow',
        'marker': 'H',
    },
    {
        'name': 'Numerical - v1 (Muon lr=0.01)',
        'file': '000_336f8548-41ba-4707-b415-f67cd37a7a74.txt',
        'train_color': 'red',
        'val_color': 'red',
        'marker': 'H',
    },
    {
        'name': 'Numerical - v2 (Muon lr=0.02)',
        'file': 'hyperball_numerical_2.out',
        'train_color': 'purple',
        'val_color': 'purple',
        'marker': 'x',
    },
    {
        'name': 'Record 7',
        'file': 'record_7.txt',
        'train_color': 'black',
        'val_color': 'black',
        'marker': 'H',
    },
    {
        'name': 'Numerical - v3 (Muon lr=0.018, Zero-step lr=0.06)',
        'file': 'hyperball_numerical_3.out',
        'train_color': 'lime',
        'val_color': 'lime',
        'marker': 'P',
    }
]

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

def extract_val_only(filepath):
    """Extract only validation loss from log file."""
    val_steps = []
    val_loss = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if 'val_loss:' in line and 'step:' in line:
                match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    val_steps.append(step)
                    val_loss.append(loss)
    
    return val_steps, val_loss

# Create the plot
fig, ax = plt.subplots(figsize=(18, 10))

# Plot all runs
print("Extracting data from runs...")
for run in runs:
    filepath = log_dir / run['file']
    
    train_steps, train_loss, val_steps, val_loss = extract_training_data(filepath)
    
    if not train_steps and not val_steps:
        print(f"  {run['name']}: No data yet (still compiling or not started)")
        continue
    
    if train_steps:
        # Plot train loss (lighter, thinner)
        # Make Hyperball (Muon lr=0.01) -switch train loss thicker and more visible
        train_linewidth = 3 if run['name'] == 'Hyperball (Muon lr=0.01) -switch' else 1
        train_alpha = 0.2 if run['name'] == 'Hyperball (Muon lr=0.01) -switch' else 0.2
        ax.plot(train_steps, train_loss, color=run['train_color'], alpha=train_alpha, linewidth=train_linewidth, 
                label=f"{run['name']} - Train")
    
    if val_steps:
        # Plot val loss (darker, thicker, with markers)
        ax.plot(val_steps, val_loss, color=run['val_color'], linewidth=2.5, 
                marker=run['marker'], markersize=5, label=f"{run['name']} - Val")
        
        # Print stats
        print(f"  {run['name']}:")
        if train_loss:
            print(f"    Train: {len(train_steps)} points, Final: {train_loss[-1]:.4f}")
        if val_loss:
            print(f"    Val: {len(val_steps)} points, Final: {val_loss[-1]:.6f}")

# Plot previous record
print("\nExtracting data from previous record...")
prev_val_steps, prev_val_loss = extract_val_only(prev_record_path)
# ax.plot(prev_val_steps, prev_val_loss, 'r-', linewidth=2.5, marker='*', markersize=7, 
#         label='Previous Record - Val', zorder=10)
print(f"  Val: {len(prev_val_steps)} points, Final: {prev_val_loss[-1]:.6f}")

ax.set_xlabel('Step', fontsize=16)
ax.set_ylabel('Loss', fontsize=16)
ax.set_title('Training Comparison: Gemma RMSNorm Runs vs Previous Record', fontsize=18, fontweight='bold')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=14)

# Zoom to show details
ax.set_ylim([2.5, 4.0])

plt.tight_layout()

# Save figure
# output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"\nPlot saved to {output_path}")

# Print improvements
if prev_val_loss:
    prev_final = prev_val_loss[-1]
    print(f"\nComparison to Previous Record (val loss: {prev_final:.6f}):")
    for run in runs:
        filepath = log_dir / run['file']
        _, _, _, val_loss = extract_training_data(filepath)
        if val_loss:
            improvement = prev_final - val_loss[-1]
            status = "✓" if improvement > 0 else "✗"
            print(f"  {status} {run['name']}: {val_loss[-1]:.6f} (improvement: {improvement:+.6f})")
