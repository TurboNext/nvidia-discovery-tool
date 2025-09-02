#!/usr/bin/env python3

# Read the current file
with open('nvidia_discovery.py', 'r') as f:
    content = f.read()

# Replace the entire query section with a simple working one
old_query_section = '''        # Try basic query first
        query_string = ','.join(basic_query_fields)
        success, stdout, stderr = self._run_command([
            'nvidia-smi',
            f'--query-gpu={query_string}',
            '--format=csv,noheader,nounits'
        ])
        
        if not success:
            self.logger.error(f"Failed to query GPU information: {stderr}")
            # Try even simpler query
            simple_query = 'index,name,driver_version,memory.total,memory.used'
            success, stdout, stderr = self._run_command([
                'nvidia-smi',
                f'--query-gpu={simple_query}',
                '--format=csv,noheader,nounits'
            ])
            if not success:
                self.logger.error(f"Failed with simple query too: {stderr}")
                return []'''

new_query_section = '''        # Use simple working query
        simple_query = 'index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id'
        success, stdout, stderr = self._run_command([
            'nvidia-smi',
            f'--query-gpu={simple_query}',
            '--format=csv,noheader,nounits'
        ])
        
        if not success:
            self.logger.error(f"Failed to query GPU information: {stderr}")
            return []'''

content = content.replace(old_query_section, new_query_section)

# Write back
with open('nvidia_discovery.py', 'w') as f:
    f.write(content)

print("Fixed nvidia_discovery.py - using simple working query")
