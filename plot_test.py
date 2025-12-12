import json
import re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# File paths
log_dir = Path(__file__).parent / 'logs'
output_path = Path(__file__).parent / 'test.pdf'

# Define runs with norm logging
runs = [
    {
        'name': 'Origin (Hyperball lr=0.025)',
        'log_file': '000_91a27ca8-3c5a-4c0f-923f-708894f1951d.txt',
        'norm_file': '000_91a27ca8-3c5a-4c0f-923f-708894f1951d_norm_log.json',
        'color': '#2563eb',  # blue
        'linestyle': '-',
        'linewidth': 2,
    },
]

def load_norm_data(filepath):
    """Load norm data from JSON/JSONL file."""
    data = []

    if not filepath.exists():
        return []

    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            f.seek(0)

            if first_line.strip().startswith('['):
                content = f.read()
                try:
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
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
    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return []

print("Loading data...")

# Load data for all runs
runs_data = []
for run in runs:
    print(f"\n  Loading {run['name']}...")
    norm_file = log_dir / run['norm_file']
    if not norm_file.exists():
        norm_file_jsonl = Path(str(norm_file).replace('.json', '.jsonl'))
        if norm_file_jsonl.exists():
            norm_file = norm_file_jsonl

    norm_data = load_norm_data(norm_file)
    runs_data.append({
        'info': run,
        'norm_data': norm_data,
    })
    print(f"    Norm data: {len(norm_data)} entries")

# Analyze block 0
block_idx = 0
weight_configs = [
    {'name': 'Q', 'key': f'block{block_idx}_attn_q_w', 'color': '#10b981', 'show_fit': True},  # green
    {'name': 'K', 'key': f'block{block_idx}_attn_k_w', 'color': '#3b82f6', 'show_fit': True},  # blue
    {'name': 'V', 'key': f'block{block_idx}_attn_v_w', 'color': '#ef4444', 'show_fit': True},  # red
    {'name': 'O', 'key': f'block{block_idx}_attn_o_w', 'color': '#f59e0b', 'show_fit': False},  # orange
    {'name': 'MLP FC', 'key': f'block{block_idx}_mlp_fc_w', 'color': '#8b5cf6', 'show_fit': True},  # purple
    {'name': 'MLP Proj', 'key': f'block{block_idx}_mlp_proj_w', 'color': '#06b6d4', 'show_fit': False},  # cyan
]

# Fitting functions
def fit_curve(steps, values):
    """Fit using 1/t as features: y = a/t^3 + b/t^2 + c/t + d"""
    x = np.array(steps, dtype=float)
    y = np.array(values)

    try:
        # Create design matrix with 1/t features
        # [1/t^3, 1/t^2, 1/t, 1]
        X = np.column_stack([
            1 / (x ** 3),
            1 / (x ** 2),
            1 / x,
            np.ones_like(x)
        ])

        # Fit using least squares
        coeffs, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)
        a, b, c, d = coeffs

        # Calculate fitted values
        fitted_y = a / (x ** 3) + b / (x ** 2) + c / x + d

        # Calculate R²
        residuals = y - fitted_y
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return fitted_y, r2, (a, b, c, d)
    except:
        return None

# Create figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Plot each weight in separate subplot
for idx, weight_cfg in enumerate(weight_configs):
    ax = axes[idx]

    for run_data in runs_data:
        if not run_data['norm_data']:
            continue

        steps = []
        inv_norms = []

        for entry in run_data['norm_data']:
            if weight_cfg['key'] in entry and entry[weight_cfg['key']] > 0:
                steps.append(entry['step'])
                inv_norms.append(1.0 / entry[weight_cfg['key']])

        if steps and inv_norms:
            print(f"  {weight_cfg['name']}: {len(steps)} points, range=[{min(inv_norms):.6f}, {max(inv_norms):.6f}]")

            # Plot original data with lighter color
            ax.plot(steps, inv_norms,
                   color=weight_cfg['color'],
                   linewidth=2,
                   alpha=0.4,
                   linestyle='-',
                   label='Data')

            # Fit and plot smooth curve
            fit_result = fit_curve(steps, inv_norms)
            if fit_result is not None:
                fitted_y, r2, (a, b, c, d) = fit_result

                if weight_cfg['show_fit']:
                    # Format coefficients in scientific notation
                    def format_coeff(val):
                        if abs(val) < 1e-10:
                            return None
                        if abs(val) >= 1000 or abs(val) < 0.001:
                            return f"{val:.2e}"
                        else:
                            return f"{val:.4f}"

                    # Build formula string: y = a/t³ + b/t² + c/t + d
                    terms = []
                    a_str = format_coeff(a)
                    if a_str is not None:
                        terms.append(f"{a_str}/t³")

                    b_str = format_coeff(b)
                    if b_str is not None:
                        if len(terms) > 0 and not b_str.startswith('-'):
                            terms.append('+')
                        terms.append(f"{b_str}/t²")

                    c_str = format_coeff(c)
                    if c_str is not None:
                        if len(terms) > 0 and not c_str.startswith('-'):
                            terms.append('+')
                        terms.append(f"{c_str}/t")

                    d_str = format_coeff(d)
                    if d_str is not None:
                        if len(terms) > 0 and not d_str.startswith('-'):
                            terms.append('+')
                        terms.append(d_str)

                    formula = ''.join(terms) if terms else 'constant'

                    ax.plot(steps, fitted_y,
                           color=weight_cfg['color'],
                           linewidth=3,
                           linestyle='-',
                           alpha=0.9,
                           label=f'y={formula}\n(R²={r2:.3f})')
                else:
                    # For O and MLP Proj, just show fit without formula
                    ax.plot(steps, fitted_y,
                           color=weight_cfg['color'],
                           linewidth=3,
                           linestyle='-',
                           alpha=0.9,
                           label=f'Fit (R²={r2:.3f})')

    # Styling
    ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('1 / Frobenius Norm', fontsize=11, fontweight='bold')
    ax.set_title(f'Block {block_idx} - {weight_cfg["name"]}',
                fontsize=13, fontweight='bold', pad=10)

    # Set y-axis limit to show detail (0 to 1/10 or slightly higher)
    if len(inv_norms) > 0:
        max_val = max(inv_norms)
        if max_val < 0.2:  # Small values like Q, K, V, MLP FC
            ax.set_ylim([0, 0.1])
        else:  # Large values like O, MLP Proj
            ax.set_ylim([0, max(6, max_val * 1.1)])

    # Grid styling - clear and prominent
    ax.grid(True, which='major', alpha=0.6, linestyle='-', linewidth=0.8, color='gray')
    ax.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='gray')
    ax.minorticks_on()

    # Legend
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95, handlelength=1.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

# Main title
fig.suptitle('Block 0 Weight Inverse Norms - Hyperball Origin Run',
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"\n✓ Test plot saved to {output_path}")

import os
print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
