import json
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7

# File paths
log_dir = Path(__file__).parent / 'logs'
output_path = Path(__file__).parent / 'figure' / 'norm_dashboard.pdf'

# Define runs with norm logging
runs = [
    {
        'name': 'Origin',
        'log_file': '000_91a27ca8-3c5a-4c0f-923f-708894f1951d.txt',
        'norm_file': '000_91a27ca8-3c5a-4c0f-923f-708894f1951d_norm_log.json',  # Will try both .json and .jsonl
        'color': '#2ecc71',  # green
        'linestyle': '-',
        'linewidth': 2,
    },
    {
        'name': 'Gemma lr=0.005',
        'log_file': '000_5c54e40d-de02-4292-a0e3-1e79ca9ff302.txt',
        'norm_file': '000_5c54e40d-de02-4292-a0e3-1e79ca9ff302_norm_log.json',  # Will try both .json and .jsonl
        'color': '#e74c3c',  # red
        'linestyle': '-',
        'linewidth': 2,
    },
    {
        'name': 'Gemma lr=0.01',
        'log_file': '000_d3659f31-4462-4cba-93cd-612dc081ae78.txt',
        'norm_file': '000_d3659f31-4462-4cba-93cd-612dc081ae78_norm_log.json',  # Will try both .json and .jsonl
        'color': '#3498db',  # blue
        'linestyle': '-',
        'linewidth': 2,
    },
    {
        'name': 'Switch lr=0.01',
        'log_file': '000_b2a522c8-ddf2-40ef-b08c-22b083e4d26f.txt',
        'norm_file': '000_b2a522c8-ddf2-40ef-b08c-22b083e4d26f_norm_log.json',  # Will try both .json and .jsonl
        'color': '#9b59b6',  # purple
        'linestyle': '-',
        'linewidth': 3,
    },
]

def extract_val_loss(filepath):
    """Extract validation loss from log file."""
    val_steps = []
    val_loss = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'val_loss:' in line and 'step:' in line:
                    match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
                    if match:
                        step = int(match.group(1))
                        loss = float(match.group(2))
                        val_steps.append(step)
                        val_loss.append(loss)
    except FileNotFoundError:
        pass

    return val_steps, val_loss

def load_norm_data(filepath):
    """Load norm data from JSON/JSONL file."""
    data = []

    # Check if file exists
    if not filepath.exists():
        return []

    try:
        with open(filepath, 'r') as f:
            # Try to detect format from first line
            first_line = f.readline()
            f.seek(0)

            # If first line starts with '[', it's old JSON array format
            if first_line.strip().startswith('['):
                content = f.read()
                try:
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
                    # Try to fix incomplete JSON
                    last_complete = content.rfind('},\n  {')
                    if last_complete == -1:
                        last_complete = content.rfind('}')
                    if last_complete != -1:
                        fixed_content = content[:last_complete+1] + '\n]'
                        try:
                            data = json.loads(fixed_content)
                            print(f"    Warning: JSON incomplete, loaded {len(data)} entries")
                            return data
                        except:
                            pass
            else:
                # Assume JSONL format (one JSON object per line)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.JSONDecodeError:
                            print(f"    Warning: Skipping malformed line {line_num}")
                            continue
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return []

print("Loading data...")

# Load data for all runs first
runs_data = []
for run in runs:
    print(f"\n  Loading {run['name']}...")
    val_steps, val_loss = extract_val_loss(log_dir / run['log_file'])

    # Try both .json and .jsonl extensions
    norm_file = log_dir / run['norm_file']
    if not norm_file.exists():
        # Try .jsonl extension
        norm_file_jsonl = Path(str(norm_file).replace('.json', '.jsonl'))
        if norm_file_jsonl.exists():
            norm_file = norm_file_jsonl

    norm_data = load_norm_data(norm_file)
    runs_data.append({
        'info': run,
        'val_steps': val_steps,
        'val_loss': val_loss,
        'norm_data': norm_data,
    })
    print(f"    Val loss: {len(val_steps)} points")
    print(f"    Norm data: {len(norm_data)} entries")

# Determine number of layers and blocks
num_blocks = 16  # GPT model has 16 blocks
num_attn_weights = 4  # q, k, v, o
num_mlp_weights = 2  # fc, proj

# Detect actual RMSNorm keys from data
rmsnorm_keys = []
for run_data in runs_data:
    if run_data['norm_data'] and len(run_data['norm_data']) > 0:
        first_entry = run_data['norm_data'][0]
        for key in first_entry.keys():
            if 'norm' in key and 'scale' in key.lower() or key.endswith('_norm_scale'):
                if key not in rmsnorm_keys:
                    rmsnorm_keys.append(key)

# Calculate actual number of subplots needed
num_plots_section1 = 1
num_plots_mlp = num_blocks * num_mlp_weights
num_plots_qkvo = (num_blocks - 1) * num_attn_weights  # block 7 has no attention
num_plots_section2 = num_plots_mlp + num_plots_qkvo
num_plots_section3 = len(rmsnorm_keys)  # Actual RMSNorm scalers
num_plots_section4 = 6

# Create figure with actual number of subplots needed
ncols = 4
# Section 1 (loss curve) will span 2 columns, others use regular grid
# Calculate rows needed: 1 row for loss (spans 2 cols) + rows for remaining plots
remaining_plots = num_plots_section2 + num_plots_section3 + num_plots_section4
remaining_rows = (remaining_plots + ncols - 1) // ncols
nrows = 1 + remaining_rows  # 1 row for loss + remaining rows

print(f"\nCreating dashboard:")
print(f"  Section 1 (Loss): {num_plots_section1} plot (spanning 2 columns)")
print(f"  Section 2 (Matrix weights): {num_plots_section2} plots")
print(f"  Section 3 (RMSNorm): {num_plots_section3} plots")
print(f"  Section 4 (Others): {num_plots_section4} plots")
print(f"  Total figure: {nrows} rows × {ncols} cols")

fig = plt.figure(figsize=(28, 5 * nrows))
gs = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.25, top=0.98, bottom=0.02)

# ============== SECTION 1: Loss Curves ==============
print("\nPlotting Section 1: Loss Curves...")
# Loss curve spans first 2 columns of first row
ax = fig.add_subplot(gs[0, :2])

for run_data in runs_data:
    if run_data['val_steps']:
        ax.plot(run_data['val_steps'], run_data['val_loss'],
                color=run_data['info']['color'],
                linewidth=run_data['info']['linewidth'],
                linestyle=run_data['info']['linestyle'],
                label=run_data['info']['name'], alpha=0.85)

ax.set_xlabel('Training Step', fontweight='bold')
ax.set_ylabel('Validation Loss', fontweight='bold')
ax.set_title('Validation Loss Curves', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, edgecolor='black')
ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.6)
ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.4)
ax.minorticks_on()
ax.set_ylim([2.5, 4.0])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============== SECTION 2: Matrix Weights ==============
print("\nPlotting Section 2: Matrix Weight Norms...")

# Create axes for remaining plots (starting from row 0, col 2)
plot_positions = []
# Fill rest of first row (columns 2-3)
for col in range(2, ncols):
    plot_positions.append((0, col))
# Fill remaining rows
for row in range(1, nrows):
    for col in range(ncols):
        plot_positions.append((row, col))

axes_remaining = [fig.add_subplot(gs[pos[0], pos[1]]) for pos in plot_positions[:remaining_plots]]
plot_idx = 0

# Section 2a: MLP weights
print("  MLP weights...")
for block_idx in range(num_blocks):
    for weight_name in ['fc_w', 'proj_w']:
        ax = axes_remaining[plot_idx]
        plot_idx += 1

        key = f"block{block_idx}_mlp_{weight_name}"

        has_data = False
        for run_data in runs_data:
            if run_data['norm_data']:
                steps = [entry['step'] for entry in run_data['norm_data'] if key in entry]
                norms = [entry[key] for entry in run_data['norm_data'] if key in entry]

                if steps and norms:
                    has_data = True
                    ax.plot(steps, norms,
                            color=run_data['info']['color'],
                            linewidth=run_data['info']['linewidth']-0.5,
                            linestyle=run_data['info']['linestyle'],
                            alpha=0.8, label=run_data['info']['name'])

        weight_label = 'FC' if weight_name == 'fc_w' else 'Proj'
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Block {block_idx} MLP {weight_label}', fontsize=10, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.minorticks_on()
        if has_data:
            ax.legend(loc='best', framealpha=0.8, fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Section 2b: QKVO weights
print("  QKVO weights...")
attn_weight_names = ['q_w', 'k_w', 'v_w', 'o_w']
for block_idx in range(num_blocks):
    if block_idx == 7:  # Skip block 7 (no attention)
        continue

    for weight_name in attn_weight_names:
        ax = axes_remaining[plot_idx]
        plot_idx += 1

        key = f"block{block_idx}_attn_{weight_name}"

        has_data = False
        for run_data in runs_data:
            if run_data['norm_data']:
                steps = [entry['step'] for entry in run_data['norm_data'] if key in entry]
                norms = [entry[key] for entry in run_data['norm_data'] if key in entry]

                if steps and norms:
                    has_data = True
                    ax.plot(steps, norms,
                            color=run_data['info']['color'],
                            linewidth=run_data['info']['linewidth']-0.5,
                            linestyle=run_data['info']['linestyle'],
                            alpha=0.8, label=run_data['info']['name'])

        weight_label = weight_name.replace('_w', '').upper()
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Block {block_idx} Attn {weight_label}', fontsize=10, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.minorticks_on()
        if has_data:
            ax.legend(loc='best', framealpha=0.8, fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# ============== SECTION 3: RMSNorm Scalers ==============
print("\nPlotting Section 3: RMSNorm Scalers...")

if rmsnorm_keys:
    print(f"  Plotting {len(rmsnorm_keys)} RMSNorm scaler keys")
    for idx, key in enumerate(rmsnorm_keys):
        ax = axes_remaining[plot_idx]
        plot_idx += 1

        has_data = False
        for run_data in runs_data:
            if run_data['norm_data']:
                steps = [entry['step'] for entry in run_data['norm_data'] if key in entry]
                norms = [entry[key] for entry in run_data['norm_data'] if key in entry]

                if steps and norms:
                    has_data = True
                    ax.plot(steps, norms,
                            color=run_data['info']['color'],
                            linewidth=run_data['info']['linewidth']-0.5,
                            linestyle=run_data['info']['linestyle'],
                            alpha=0.8, label=run_data['info']['name'])

        ax.set_xlabel('Step')
        ax.set_ylabel('Scale Parameter')
        ax.set_title(f'RMSNorm: {key}', fontsize=10, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.minorticks_on()
        if has_data:
            ax.legend(loc='best', framealpha=0.8, fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
else:
    print("  No RMSNorm scalers found")

# ============== SECTION 4: Others ==============
print("\nPlotting Section 4: Other Parameters...")

other_keys = ['embed_w', 'value_embed0_w', 'value_embed1_w', 'value_embed2_w', 'lm_head_w', 'scalars']

for idx, key in enumerate(other_keys):
    ax = axes_remaining[plot_idx]
    plot_idx += 1

    has_data = False
    for run_data in runs_data:
        if run_data['norm_data']:
            steps = [entry['step'] for entry in run_data['norm_data'] if key in entry]
            norms = [entry[key] for entry in run_data['norm_data'] if key in entry]

            if steps and norms:
                has_data = True
                ax.plot(steps, norms,
                        color=run_data['info']['color'],
                        linewidth=run_data['info']['linewidth']-0.5,
                        linestyle=run_data['info']['linestyle'],
                        alpha=0.8, label=run_data['info']['name'])

    clean_key = key.replace('_w', '').replace('_', ' ').title()
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title(f'{clean_key}', fontsize=10, fontweight='bold')
    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax.minorticks_on()
    if has_data:
        ax.legend(loc='best', framealpha=0.8, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide unused subplots in remaining axes
for idx in range(plot_idx, len(axes_remaining)):
    axes_remaining[idx].axis('off')

# Add main title at the very top
fig.suptitle('Parameter Norm Dashboard - Origin vs Gemma RMSNorm',
             fontsize=18, fontweight='bold', y=0.995)

# Save figure
output_path.parent.mkdir(exist_ok=True)
print(f"\nSaving figure to {output_path}...")
try:
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    import os
    file_size = os.path.getsize(output_path)
    if file_size == 0:
        print("  WARNING: PDF file is 0 bytes!")
        # Try PNG instead
        output_path_png = output_path.with_suffix('.png')
        print(f"  Trying PNG instead: {output_path_png}")
        plt.savefig(output_path_png, format='png', dpi=100, bbox_inches='tight')
        print(f"  PNG saved: {os.path.getsize(output_path_png)} bytes")
    else:
        print(f"✓ Dashboard saved to {output_path}")
        print(f"  File size: {file_size / 1024:.1f} KB")
except Exception as e:
    print(f"  ERROR saving figure: {e}")
    import traceback
    traceback.print_exc()

print(f"  Total subplots: {plot_idx}")
print(f"  Figure size: {fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.1f} inches")
