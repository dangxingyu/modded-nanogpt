#!/usr/bin/env python3
"""Extract sweep results from log files."""
import os
import re
from pathlib import Path

logs_dir = Path("logs")
results = []

for log_file in sorted(logs_dir.glob("000_*.txt")):
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Extract hyperparameters
        zero_lr_match = re.search(r'zero_step_lr = ([\d.]+)', content)
        muon_lr_match = re.search(r'muon_lr = ([\d.]+)', content)
        cooldown_match = re.search(r'cooldown_frac = ([\d.]+)', content)

        # Extract final validation loss
        val_loss_matches = re.findall(r'val_loss:([\d.]+)', content)

        if zero_lr_match and muon_lr_match and val_loss_matches:
            zero_lr = float(zero_lr_match.group(1))
            muon_lr = float(muon_lr_match.group(1))
            cooldown = float(cooldown_match.group(1)) if cooldown_match else 0.6
            final_val_loss = float(val_loss_matches[-1])

            results.append({
                'file': log_file.name,
                'zero_lr': zero_lr,
                'muon_lr': muon_lr,
                'cooldown': cooldown,
                'val_loss': final_val_loss
            })
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

# Sort by val_loss
results.sort(key=lambda x: x['val_loss'])

print(f"Found {len(results)} sweep runs:\n")
print(f"{'Rank':<5} {'File':<45} {'Zero LR':<10} {'Muon LR':<10} {'Cooldown':<10} {'Val Loss':<10}")
print("-" * 100)
for i, r in enumerate(results, 1):
    print(f"{i:<5} {r['file']:<45} {r['zero_lr']:<10.4f} {r['muon_lr']:<10.4f} {r['cooldown']:<10.2f} {r['val_loss']:<10.6f}")

print(f"\n\nBest run: {results[0]['file']}")
print(f"  zero_step_lr={results[0]['zero_lr']}, muon_lr={results[0]['muon_lr']}, cooldown={results[0]['cooldown']}")
print(f"  val_loss={results[0]['val_loss']}")
