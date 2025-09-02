#!/usr/bin/env python3
import re

# Read the current file
with open('nvidia_discovery.py', 'r') as f:
    content = f.read()

# Replace the problematic query with a working one
old_query = "--query-gpu=index,name,driver_version,cuda_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id"

new_query = "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id"

# Replace in the file
content = content.replace(old_query, new_query)

# Write back
with open('nvidia_discovery.py', 'w') as f:
    f.write(content)

print("Fixed nvidia_discovery.py - removed invalid cuda_version field")
