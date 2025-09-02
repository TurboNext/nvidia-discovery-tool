#!/usr/bin/env python3
import subprocess

def test_gpu_parsing():
    # Run the exact command that works
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id',
        '--format=csv,noheader,nounits'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    print(f"Return code: {result.returncode}")
    print(f"Stdout:\n{result.stdout}")
    print(f"Stderr: {result.stderr}")
    
    # Test parsing
    lines = result.stdout.strip().split('\n')
    print(f"\nNumber of lines: {len(lines)}")
    
    for i, line in enumerate(lines):
        print(f"Line {i}: {line}")
        parts = [part.strip() for part in line.split(',')]
        print(f"  Parts: {len(parts)} - {parts}")

if __name__ == "__main__":
    test_gpu_parsing()
