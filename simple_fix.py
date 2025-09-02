#!/usr/bin/env python3

# Read the current file
with open('nvidia_discovery.py', 'r') as f:
    content = f.read()

# Find and replace the problematic query section
# Look for the basic_query_fields section and replace it
old_section = '''        # Get comprehensive GPU information - try with basic fields first
        basic_query_fields = [
            'index', 'name', 'driver_version', 'cuda_version',
            'memory.total', 'memory.used', 'memory.free',
            'utilization.gpu', 'utilization.memory', 'temperature.gpu',
            'power.draw', 'power.limit', 'uuid', 'pci.bus_id'
        ]'''

new_section = '''        # Get comprehensive GPU information - use only working fields
        basic_query_fields = [
            'index', 'name', 'driver_version',
            'memory.total', 'memory.used', 'memory.free',
            'utilization.gpu', 'utilization.memory', 'temperature.gpu',
            'power.draw', 'power.limit', 'uuid', 'pci.bus_id'
        ]'''

content = content.replace(old_section, new_section)

# Also fix the simple query fallback
old_simple = "simple_query = 'index,name,driver_version,memory.total,memory.used'"
new_simple = "simple_query = 'index,name,driver_version,memory.total,memory.used,memory.free'"

content = content.replace(old_simple, new_simple)

# Write back
with open('nvidia_discovery.py', 'w') as f:
    f.write(content)

print("Fixed nvidia_discovery.py - using only working fields")
