#!/usr/bin/env python3

# Read the current file
with open('nvidia_discovery.py', 'r') as f:
    content = f.read()

# Find the discover_gpus method and replace it entirely
old_method = '''    def discover_gpus(self) -> List[GPUInfo]:
        """Discover all NVIDIA GPUs in the system"""
        self.logger.info("Discovering NVIDIA GPUs...")
        
        # Check if nvidia-smi is available
        success, _, _ = self._run_command(['nvidia-smi', '--version'])
        if not success:
            self.logger.error("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
            return []
        
        # Get comprehensive GPU information - use only working fields
        basic_query_fields = [
            'index', 'name', 'driver_version',
            'memory.total', 'memory.used', 'memory.free',
            'utilization.gpu', 'utilization.memory', 'temperature.gpu',
            'power.draw', 'power.limit', 'uuid', 'pci.bus_id'
        ]
        
        # Use simple working query
        simple_query = 'index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id'
        success, stdout, stderr = self._run_command([
            'nvidia-smi',
            f'--query-gpu={simple_query}',
            '--format=csv,noheader,nounits'
        ])
        
        if not success:
            self.logger.error(f"Failed to query GPU information: {stderr}")
            return []
        
        gpus = self._parse_gpu_info(stdout)
        self.logger.info(f"Discovered {len(gpus)} GPU(s)")
        
        return gpus'''

new_method = '''    def discover_gpus(self) -> List[GPUInfo]:
        """Discover all NVIDIA GPUs in the system"""
        self.logger.info("Discovering NVIDIA GPUs...")
        
        # Check if nvidia-smi is available
        success, _, _ = self._run_command(['nvidia-smi', '--version'])
        if not success:
            self.logger.error("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
            return []
        
        # Use the exact working query
        success, stdout, stderr = self._run_command([
            'nvidia-smi',
            '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id',
            '--format=csv,noheader,nounits'
        ])
        
        if not success:
            self.logger.error(f"Failed to query GPU information: {stderr}")
            return []
        
        gpus = self._parse_gpu_info(stdout)
        self.logger.info(f"Discovered {len(gpus)} GPU(s)")
        
        return gpus'''

# Replace the method
content = content.replace(old_method, new_method)

# Write back
with open('nvidia_discovery.py', 'w') as f:
    f.write(content)

print("Fixed nvidia_discovery.py - replaced entire discover_gpus method")
