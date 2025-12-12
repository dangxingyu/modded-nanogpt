import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

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
norm_file = log_dir / 'record-7-norm.jsonl'
output_path = Path(__file__).parent / 'figure' / 'eff_lr_dashboard.pdf'

def load_norm_data(filepath):
    """Load norm data from JSONL file."""
    data = []
    metadata = None
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return None, []
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # First line is metadata
                    if entry.get('type') == 'metadata':
                        metadata = entry
                    elif 'step' in entry:
                        data.append(entry)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num}")
                    continue
        return metadata, data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, []

def get_lr_schedule(step, num_iterations, cooldown_frac):
    """Learning rate schedule: stable then linear decay."""
    x = step / num_iterations  # progress in training
    if x < 1 - cooldown_frac:
        return 1.0  # stable phase
    else:
        return (1 - x) / cooldown_frac  # linear decay phase

def get_param_lr_and_dim(key, model_dim=1024, optimizers_config=None):
    """Get the initial learning rate and last dimension of a parameter based on its key."""
    if optimizers_config is None:
        optimizers_config = {}
    
    adam_config = optimizers_config.get('AdamW', {})
    muon_config = optimizers_config.get('Muon', {})
    
    # MLP weights - use Muon optimizer
    if 'mlp_fc_w' in key or 'mlp_proj_w' in key:
        # fc_w shape: (4*dim, dim), so last dim is dim
        # proj_w shape: (dim, 4*dim), so last dim is 4*dim
        dim = model_dim if 'proj' in key else 4 * model_dim
        initial_lr = muon_config.get('lr', 0.025)
        return initial_lr, dim
    
    # Attention weights (q, k, v, o) - use Muon optimizer
    elif 'attn_q_w' in key or 'attn_k_w' in key or 'attn_v_w' in key or 'attn_o_w' in key:
        # qkvo_w shape: (hdim, dim), so last dim is dim
        initial_lr = muon_config.get('lr', 0.025)
        return initial_lr, model_dim
    
    # Embedding weights - use AdamW optimizer
    elif 'embed_w' in key or 'value_embed' in key:
        # embed shape: (vocab_size, model_dim), so last dim is model_dim
        embed_lr = adam_config.get('param_groups', {}).get('embed', {}).get('lr', 0.3)
        return embed_lr, model_dim
    
    # LM head - use AdamW optimizer
    elif 'lm_head_w' in key:
        # lm_head shape: (vocab_size, model_dim), so last dim is model_dim
        head_lr = adam_config.get('param_groups', {}).get('head', {}).get('lr', 1/320)
        return head_lr, model_dim
    
    # Scalars - use AdamW optimizer
    elif 'scalars' in key:
        scalar_lr = adam_config.get('param_groups', {}).get('scalar', {}).get('lr', 0.015)
        return scalar_lr, 1
    
    # RMSNorm (if present) - typically use same as embeddings
    elif 'rmsnorm' in key and 'scale' in key:
        embed_lr = adam_config.get('param_groups', {}).get('embed', {}).get('lr', 0.3)
        return embed_lr, model_dim
    
    # Default: assume Muon and model_dim
    return muon_config.get('lr', 0.025), model_dim

print("Loading data...")
metadata, norm_data = load_norm_data(norm_file)

if metadata is None:
    print("Error: Could not load metadata. Using default values.")
    model_dim = 1024
    num_layers = 16
    num_iterations = 6450
    cooldown_frac = 0.6
    optimizers_config = {}
else:
    model_dim = metadata.get('model', {}).get('model_dim', 1024)
    num_layers = metadata.get('model', {}).get('num_layers', 16)
    num_iterations = metadata.get('hyperparameters', {}).get('num_iterations', 6450)
    cooldown_frac = metadata.get('hyperparameters', {}).get('cooldown_frac', 0.6)
    optimizers_config = metadata.get('optimizers', {})
    print(f"Model config: model_dim={model_dim}, num_layers={num_layers}")
    print(f"Training config: num_iterations={num_iterations}, cooldown_frac={cooldown_frac}")

if not norm_data:
    print("Error: No norm data found!")
    exit(1)

print(f"Loaded {len(norm_data)} steps")

# Calculate effective learning rate: lr * sqrt(w.shape[-1]) / w.norm
# where lr is the scheduled learning rate at that step
eff_lr_data = {}
min_step = 1  # Start from step 1

for entry in norm_data:
    step = entry.get('step', 0)
    if step < min_step:
        continue  # Skip steps before min_step
    
    if step not in eff_lr_data:
        eff_lr_data[step] = {}
    
    # Get current learning rate multiplier from schedule
    lr_multiplier = get_lr_schedule(step, num_iterations, cooldown_frac)
    
    for key, norm_value in entry.items():
        if key == 'step' or key == 'train_loss':
            continue
        
        if norm_value == 0:
            eff_lr_data[step][key] = 0.0
        else:
            initial_lr, dim = get_param_lr_and_dim(key, model_dim, optimizers_config)
            current_lr = initial_lr * lr_multiplier
            eff_lr = current_lr * np.sqrt(dim) / norm_value
            eff_lr_data[step][key] = eff_lr

# Extract block0_attn_q_w eff_lr data and save to JSON
print("\nExtracting block0_attn_q_w eff_lr data...")
q_key = 'block0_attn_q_w'
q_steps = []
q_eff_lr = []

for step in sorted(eff_lr_data.keys()):
    if q_key in eff_lr_data[step]:
        q_steps.append(step)
        q_eff_lr.append(eff_lr_data[step][q_key])

# Fit only from step 100 onwards
fit_start_step = 100
fit_start_idx = next((i for i, s in enumerate(q_steps) if s >= fit_start_step), 0)
print(f"  Fitting from step {fit_start_step} (index {fit_start_idx}) onwards")

q_data = {
    'steps': q_steps,
    'eff_lr': q_eff_lr,
    'num_points': len(q_steps)
}

q_output_file = Path(__file__).parent / 'q_eff_lr.json'
with open(q_output_file, 'w') as f:
    json.dump(q_data, f, indent=2)
print(f"Saved {len(q_steps)} data points to {q_output_file}")

# Fit piecewise function: first part is 1/t polynomial, second part is linear
print("\nFitting piecewise function...")

# Try to find a reasonable breakpoint (e.g., where lr schedule changes)
stable_end_step = int(num_iterations * (1 - cooldown_frac))
# Find the step index closest to stable_end_step
breakpoint_idx = min(len(q_steps) - 1, max(1, np.argmin(np.abs(np.array(q_steps) - stable_end_step))))
breakpoint_guess = q_steps[breakpoint_idx]

print(f"  Using breakpoint guess: {breakpoint_guess} (step {breakpoint_idx}/{len(q_steps)})")

# Fit first part (1/t polynomial) - simpler form: a/t + b
def first_part_func(t, a, b):
    """Simple 1/t form: a/t + b"""
    return a / np.array(t) + b

# Fit second part (linear)
def second_part_func(t, a, b):
    """Linear: a*t + b"""
    return a * np.array(t) + b

try:
    # Use only data from fit_start_step onwards for fitting
    q_steps_fit = np.array(q_steps)[fit_start_idx:]
    q_eff_lr_fit = np.array(q_eff_lr)[fit_start_idx:]
    
    # Split data at breakpoint (only for fitting data)
    first_mask_fit = q_steps_fit < breakpoint_guess
    second_mask_fit = q_steps_fit >= breakpoint_guess
    
    first_steps_fit = q_steps_fit[first_mask_fit]
    first_eff_lr_fit = q_eff_lr_fit[first_mask_fit]
    second_steps_fit = q_steps_fit[second_mask_fit]
    second_eff_lr_fit = q_eff_lr_fit[second_mask_fit]
    
    print(f"  First part (for fitting): {len(first_steps_fit)} points")
    print(f"  Second part (for fitting): {len(second_steps_fit)} points")
    
    # Fit first part: a/t + b
    if len(first_steps_fit) > 2:
        # Use 1/t as feature
        X1 = np.column_stack([1.0 / first_steps_fit, np.ones_like(first_steps_fit)])
        coeffs1, residuals1, rank1, s1 = np.linalg.lstsq(X1, first_eff_lr_fit, rcond=None)
        a1, b1 = coeffs1
    else:
        a1, b1 = 0.0, np.mean(first_eff_lr_fit) if len(first_eff_lr_fit) > 0 else 0.0
    
    # Fit second part: linear
    if len(second_steps_fit) > 1:
        X2 = np.column_stack([second_steps_fit, np.ones_like(second_steps_fit)])
        coeffs2, residuals2, rank2, s2 = np.linalg.lstsq(X2, second_eff_lr_fit, rcond=None)
        a2, b2 = coeffs2
    else:
        a2, b2 = 0.0, np.mean(second_eff_lr_fit) if len(second_eff_lr_fit) > 0 else 0.0
    
    # Now compute fitted values for ALL steps (including before fit_start_step)
    q_steps_all = np.array(q_steps)
    q_fitted = np.zeros_like(q_eff_lr)
    
    # Apply fitted functions to all steps
    first_mask_all = q_steps_all < breakpoint_guess
    second_mask_all = q_steps_all >= breakpoint_guess
    
    # First part: apply min(拟合值, 0.05)
    first_fitted_raw = a1 / q_steps_all[first_mask_all] + b1
    q_fitted[first_mask_all] = np.minimum(first_fitted_raw, 0.05)
    q_fitted[second_mask_all] = a2 * q_steps_all[second_mask_all] + b2
    
    # Calculate R² only on fitting data
    residuals_fit = q_eff_lr_fit - q_fitted[fit_start_idx:]
    ss_res = np.sum(residuals_fit ** 2)
    ss_tot = np.sum((q_eff_lr_fit - np.mean(q_eff_lr_fit)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Save fit results
    fit_results = {
        'breakpoint': int(breakpoint_guess),
        'fit_start_step': int(fit_start_step),
        'first_part': {
            'type': '1_over_t_linear_capped',
            'formula': f'min({a1:.6e}/t + {b1:.6e}, 0.05)',
            'coefficients': {
                'a': float(a1),
                'b': float(b1),
                'cap': 0.05
            },
            'num_points': int(len(first_steps_fit))
        },
        'second_part': {
            'type': 'linear',
            'formula': f'{a2:.6e} * t + {b2:.6e}',
            'coefficients': {
                'a': float(a2),
                'b': float(b2)
            },
            'num_points': int(len(second_steps_fit))
        },
        'r_squared': float(r2),
        'fitted_values': q_fitted.tolist()
    }
    
    # Add fit results to q_data
    q_data['fit'] = fit_results
    
    # Save updated data with fit
    with open(q_output_file, 'w') as f:
        json.dump(q_data, f, indent=2)
    
    print(f"Fit successful!")
    print(f"  Breakpoint: {breakpoint_guess}")
    print(f"  First part (t < {breakpoint_guess}): min({a1:.6e}/t + {b1:.6e}, 0.05)")
    print(f"  Second part (t >= {breakpoint_guess}): {a2:.6e} * t + {b2:.6e}")
    print(f"  R² = {r2:.6f}")
    
except Exception as e:
    print(f"Fit failed: {e}")
    import traceback
    traceback.print_exc()

# Determine number of layers and blocks
num_blocks = num_layers
num_attn_weights = 4  # q, k, v, o
num_mlp_weights = 2  # fc, proj

# Detect actual parameter keys from data
all_keys = set()
for step_data in eff_lr_data.values():
    all_keys.update(step_data.keys())

# Detect RMSNorm keys
rmsnorm_keys = []
for key in all_keys:
    if 'norm' in key and 'scale' in key.lower() or key.endswith('_norm_scale'):
        if key not in rmsnorm_keys:
            rmsnorm_keys.append(key)

# Calculate actual number of subplots needed
num_plots_section1 = 1  # Loss curve
num_plots_mlp = num_blocks * num_mlp_weights
num_plots_qkvo = (num_blocks - 1) * num_attn_weights  # block 7 has no attention (if applicable)
# Check if block 7 has attention
has_block7_attn = any(f'block7_attn' in key for key in all_keys)
if has_block7_attn:
    num_plots_qkvo = num_blocks * num_attn_weights
num_plots_section2 = num_plots_mlp + num_plots_qkvo
num_plots_section3 = len(rmsnorm_keys)  # Actual RMSNorm scalers
# Count actual other keys
other_keys_count = 0
for key in all_keys:
    if key in ['step', 'train_loss']:
        continue
    if 'mlp' in key or 'attn' in key:
        continue
    if 'rmsnorm' in key or 'norm' in key:
        continue
    other_keys_count += 1
num_plots_section4 = other_keys_count

# Create figure with actual number of subplots needed
ncols = 4
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
ax = fig.add_subplot(gs[0, :2])

# Get train_loss from original norm_data (only from min_step onwards)
train_loss_steps = []
train_loss_values = []
for entry in norm_data:
    step = entry.get('step', 0)
    if step >= min_step and 'train_loss' in entry:
        train_loss_steps.append(step)
        train_loss_values.append(entry['train_loss'])

if train_loss_steps:
    ax.plot(train_loss_steps, train_loss_values, color='#2ecc71', linewidth=2, linestyle='-', alpha=0.85, label='Train Loss')

ax.set_xlabel('Training Step', fontweight='bold')
ax.set_ylabel('Train Loss', fontweight='bold')
ax.set_title('Training Loss Curve', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.9, edgecolor='black')
ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.6)
ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.4)
ax.minorticks_on()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============== SECTION 2: Matrix Weights ==============
print("\nPlotting Section 2: Matrix Weight Effective LR...")

# Create axes for remaining plots
plot_positions = []
for col in range(2, ncols):
    plot_positions.append((0, col))
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
        steps_list = []
        eff_lr_list = []
        
        for step in sorted(eff_lr_data.keys()):
            if key in eff_lr_data[step]:
                steps_list.append(step)
                eff_lr_list.append(eff_lr_data[step][key])
                has_data = True

        if has_data and steps_list:
            ax.plot(steps_list, eff_lr_list, color='#2ecc71', linewidth=1.5, linestyle='-', alpha=0.8)

        weight_label = 'FC' if weight_name == 'fc_w' else 'Proj'
        ax.set_xlabel('Step')
        ax.set_ylabel('Effective LR')
        ax.set_title(f'Block {block_idx} MLP {weight_label}', fontsize=10, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.minorticks_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Section 2b: QKVO weights
print("  QKVO weights...")
attn_weight_names = ['q_w', 'k_w', 'v_w', 'o_w']
for block_idx in range(num_blocks):
    # Check if this block has attention
    has_attn = any(f'block{block_idx}_attn' in key for key in all_keys)
    if not has_attn:
        # Skip this block's attention plots
        for _ in attn_weight_names:
            ax = axes_remaining[plot_idx]
            plot_idx += 1
            ax.axis('off')
        continue

    for weight_name in attn_weight_names:
        ax = axes_remaining[plot_idx]
        plot_idx += 1

        key = f"block{block_idx}_attn_{weight_name}"

        has_data = False
        steps_list = []
        eff_lr_list = []
        
        for step in sorted(eff_lr_data.keys()):
            if key in eff_lr_data[step]:
                steps_list.append(step)
                eff_lr_list.append(eff_lr_data[step][key])
                has_data = True

        if has_data and steps_list:
            ax.plot(steps_list, eff_lr_list, color='#2ecc71', linewidth=1.5, linestyle='-', alpha=0.8)

        weight_label = weight_name.replace('_w', '').upper()
        ax.set_xlabel('Step')
        ax.set_ylabel('Effective LR')
        ax.set_title(f'Block {block_idx} Attn {weight_label}', fontsize=10, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.minorticks_on()
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
        steps_list = []
        eff_lr_list = []
        
        for step in sorted(eff_lr_data.keys()):
            if key in eff_lr_data[step]:
                steps_list.append(step)
                eff_lr_list.append(eff_lr_data[step][key])
                has_data = True

        if has_data and steps_list:
            ax.plot(steps_list, eff_lr_list, color='#2ecc71', linewidth=1.5, linestyle='-', alpha=0.8)

        ax.set_xlabel('Step')
        ax.set_ylabel('Effective LR')
        ax.set_title(f'RMSNorm: {key}', fontsize=10, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.minorticks_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
else:
    print("  No RMSNorm scalers found")

# ============== SECTION 4: Others ==============
print("\nPlotting Section 4: Other Parameters...")

# Find all other keys (not MLP, not attention, not RMSNorm)
other_keys = []
for key in all_keys:
    if key in ['step', 'train_loss']:
        continue
    if 'mlp' in key or 'attn' in key:
        continue
    if 'rmsnorm' in key or 'norm' in key:
        continue
    if key not in other_keys:
        other_keys.append(key)

# Sort for consistent ordering
other_keys = sorted(other_keys)

# Only plot up to available axes
for idx, key in enumerate(other_keys):
    if plot_idx >= len(axes_remaining):
        print(f"  Warning: Not enough axes for all other parameters. Stopping at {idx}/{len(other_keys)}")
        break
    
    ax = axes_remaining[plot_idx]
    plot_idx += 1

    has_data = False
    steps_list = []
    eff_lr_list = []
    
    for step in sorted(eff_lr_data.keys()):
        if key in eff_lr_data[step]:
            steps_list.append(step)
            eff_lr_list.append(eff_lr_data[step][key])
            has_data = True

    if has_data and steps_list:
        ax.plot(steps_list, eff_lr_list, color='#2ecc71', linewidth=1.5, linestyle='-', alpha=0.8)

    clean_key = key.replace('_w', '').replace('_', ' ').title()
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective LR')
    ax.set_title(f'{clean_key}', fontsize=10, fontweight='bold')
    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide unused subplots
for idx in range(plot_idx, len(axes_remaining)):
    axes_remaining[idx].axis('off')

# Add main title
fig.suptitle('Effective Learning Rate Dashboard (lr * sqrt(w.shape[-1]) / w.norm)',
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

