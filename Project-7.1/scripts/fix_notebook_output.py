#!/usr/bin/env python3
"""Update ablation notebook with warning suppression and real-time output."""

import json

def main():
    with open('ablation-dual-workers.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Update both worker script cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Add warning suppression to worker scripts
            if '%%writefile worker_gpu' in source and 'warnings.filterwarnings' not in source:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if "os.environ['CUDA_VISIBLE_DEVICES']" in line:
                        new_source.extend([
                            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n",
                            "import warnings\n",
                            "warnings.filterwarnings('ignore')\n",
                            "import logging\n",
                            "logging.getLogger('transformers').setLevel(logging.ERROR)\n",
                            "logging.getLogger('tensorflow').setLevel(logging.ERROR)\n",
                        ])
                cell['source'] = new_source
            
            # Update the subprocess cell for real-time output
            if 'subprocess.Popen' in source:
                cell['source'] = [
'''## 2. Run Both Workers in Parallel (Real-time Output)
import subprocess
import sys
import threading
import time

def stream_output(proc, name):
    """Stream process output in real-time."""
    for line in iter(proc.stdout.readline, ''):
        if line:
            print(f"{line}", end='', flush=True)
    proc.stdout.close()

print("Starting 2 workers in parallel...")
print("="*60)

# Start both processes with stdout pipe
p0 = subprocess.Popen(
    ['python', '-u', 'worker_gpu0.py'],  # -u for unbuffered
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)
p1 = subprocess.Popen(
    ['python', '-u', 'worker_gpu1.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Create threads to stream output
t0 = threading.Thread(target=stream_output, args=(p0, 'GPU0'))
t1 = threading.Thread(target=stream_output, args=(p1, 'GPU1'))

t0.start()
t1.start()

# Wait for completion
p0.wait()
p1.wait()
t0.join()
t1.join()

print("="*60)
print("Both workers finished!")
'''
                ]
    
    with open('ablation-dual-workers.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Updated notebook with:')
    print('  - Warning suppression (TF, transformers, general)')
    print('  - Real-time output streaming')

if __name__ == '__main__':
    main()
