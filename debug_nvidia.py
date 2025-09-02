#!/usr/bin/env python3
import subprocess
import sys

def test_nvidia_smi():
    print("Testing nvidia-smi queries...")
    
    # Test 1: Basic query
    print("\n1. Testing basic query:")
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,driver_version,memory.total,memory.used',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Comprehensive query
    print("\n2. Testing comprehensive query:")
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,driver_version,cuda_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Simple nvidia-smi
    print("\n3. Testing simple nvidia-smi:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_nvidia_smi()
