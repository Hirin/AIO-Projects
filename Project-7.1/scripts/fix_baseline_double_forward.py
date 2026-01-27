#!/usr/bin/env python3
"""Fix double forward pass bug in baseline worker."""

import json

with open('ablation-8frames.ipynb', 'r') as f:
    nb = json.load(f)

# Fix baseline worker
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '%%writefile worker_8frames_baseline.py' in source:
            # Find and fix the double forward pass bug
            new_source = []
            i = 0
            while i < len(cell['source']):
                line = cell['source'][i]
                # Find the buggy lines
                if 'loss = F.cross_entropy(m(x).logits, y)' in line:
                    # Replace with proper version
                    new_source.append('            logits = m(x).logits\n')
                    new_source.append('            loss = F.cross_entropy(logits, y)\n')
                    i += 1
                    # Skip the next line which has the second forward pass
                    if i < len(cell['source']) and 'm(x).logits.argmax(1)' in cell['source'][i]:
                        new_source.append('        cor += (logits.argmax(1)==y).sum().item()\n')
                        i += 1
                else:
                    new_source.append(line)
                    i += 1
            cell['source'] = new_source

with open('ablation-8frames.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('✓ Fixed double forward pass bug')
print('  Before: loss = F.cross_entropy(m(x).logits, y) + cor += (m(x).logits...)')
print('  After:  logits = m(x).logits (once!) then reuse for loss and accuracy')
print('')
print('VRAM should now be ~6-7GB for both workers (instead of 13.6GB for baseline)')
