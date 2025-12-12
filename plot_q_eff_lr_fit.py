import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Load data
data_file = Path(__file__).parent / 'q_eff_lr.json'
output_path = Path(__file__).parent / 'figure' / 'q_eff_lr_fit.pdf'

with open(data_file, 'r') as f:
    data = json.load(f)

steps = np.array(data['steps'])
eff_lr = np.array(data['eff_lr'])
fit_data = data['fit']

# Get fitted values
fitted_values = np.array(fit_data['fitted_values'])

# Print first 50 steps
print("\n" + "="*80)
print("Gold Eff LR and Predicted Eff LR for First 50 Steps:")
print("="*80)
print(f"{'Step':<10} {'Gold Eff LR':<20} {'Predicted Eff LR':<20} {'Difference':<20}")
print("-"*80)

# Print first 50 steps
for i in range(min(50, len(steps))):
    step = steps[i]
    gold = eff_lr[i]
    pred = fitted_values[i]
    diff = gold - pred
    print(f"{step:<10} {gold:<20.10e} {pred:<20.10e} {diff:<20.10e}")

print("="*80)
print(f"Total steps: {len(steps)}")
print(f"Steps range: {steps[0]} to {steps[-1]}")
print(f"Note: Fit was performed from step 100 onwards")
print("="*80 + "\n")

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot original data
ax.plot(steps, eff_lr, 
        color='#3498db', 
        linewidth=1.5, 
        alpha=0.7, 
        label='Original Data',
        zorder=1)

# Plot fitted curve
ax.plot(steps, fitted_values,
        color='#e74c3c',
        linewidth=2.5,
        alpha=0.9,
        label='Fitted Curve',
        zorder=2)

# Mark breakpoint
breakpoint = fit_data['breakpoint']
breakpoint_idx = np.argmin(np.abs(steps - breakpoint))
breakpoint_value = eff_lr[breakpoint_idx]

ax.axvline(breakpoint, 
           color='#9b59b6', 
           linestyle='--', 
           linewidth=2, 
           alpha=0.7,
           label=f'Breakpoint: {breakpoint}',
           zorder=3)
ax.plot(breakpoint, breakpoint_value, 
        'o', 
        color='#9b59b6', 
        markersize=10, 
        zorder=4)

# Add text annotations for the two parts
first_formula = fit_data['first_part']['formula']
second_formula = fit_data['second_part']['formula']
r2 = fit_data['r_squared']

# Add text box with fit information
textstr = f'Fit Results:\n'
textstr += f'First part (t < {breakpoint}):\n  {first_formula}\n'
textstr += f'Second part (t ≥ {breakpoint}):\n  {second_formula}\n'
textstr += f'R² = {r2:.6f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

# Labels and title
ax.set_xlabel('Training Step', fontweight='bold', fontsize=12)
ax.set_ylabel('Effective Learning Rate', fontweight='bold', fontsize=12)
ax.set_title('Block 0 Attention Q Weight Effective LR Fit', 
             fontsize=14, fontweight='bold', pad=15)

# Grid
ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
ax.minorticks_on()

# Legend
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save figure
output_path.parent.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
print(f"✓ Figure saved to {output_path}")

# Also save as PNG
png_path = output_path.with_suffix('.png')
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
print(f"✓ Figure saved to {png_path}")

plt.close()

