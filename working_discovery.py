#!/usr/bin/env python3
import subprocess
import sys

def discover_gpus():
    print("Discovering NVIDIA GPUs...")
    
    # Use the exact command that works
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id',
        '--format=csv,noheader,nounits'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return []
        
        lines = result.stdout.strip().split('\n')
        gpus = []
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 13:
                gpu_info = {
                    'index': parts[0],
                    'name': parts[1],
                    'driver_version': parts[2],
                    'memory_total': parts[3],
                    'memory_used': parts[4],
                    'memory_free': parts[5],
                    'utilization_gpu': parts[6],
                    'utilization_memory': parts[7],
                    'temperature': parts[8],
                    'power_draw': parts[9],
                    'power_limit': parts[10],
                    'uuid': parts[11],
                    'pci_bus_id': parts[12]
                }
                gpus.append(gpu_info)
        
        return gpus
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    gpus = discover_gpus()
    
    print(f"\nDiscovered {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']} MB")
        print(f"  Temperature: {gpu['temperature']}°C")
        print(f"  Power: {gpu['power_draw']}/{gpu['power_limit']} W")
        print(f"  UUID: {gpu['uuid']}")
        print()

if __name__ == "__main__":
    main()
