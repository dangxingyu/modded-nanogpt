#!/usr/bin/env python3
"""Test script to verify jsonlines writing works correctly."""

import json
import tempfile
from pathlib import Path

# Simulate the new logging approach
def test_jsonl_writing():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = Path(f.name)

    print(f"Testing jsonlines writing to: {temp_file}")

    # Simulate writing multiple entries
    for step in range(1, 11):
        param_norms = {
            'step': step,
            'param1': float(step * 1.5),
            'param2': float(step * 2.0),
        }

        # This is the new approach - append mode
        with open(temp_file, 'a') as f:
            f.write(json.dumps(param_norms) + '\n')

    print(f"✓ Wrote 10 entries")

    # Now read back
    print("\nReading back...")
    data = []
    with open(temp_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                entry = json.loads(line)
                data.append(entry)

    print(f"✓ Read {len(data)} entries")
    print(f"  First entry: {data[0]}")
    print(f"  Last entry: {data[-1]}")

    # Verify
    assert len(data) == 10, f"Expected 10 entries, got {len(data)}"
    assert data[0]['step'] == 1
    assert data[-1]['step'] == 10

    print("\n✓ All tests passed!")

    # Cleanup
    temp_file.unlink()
    print(f"✓ Cleaned up {temp_file}")

if __name__ == '__main__':
    test_jsonl_writing()
