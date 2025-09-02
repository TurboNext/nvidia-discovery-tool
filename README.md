# NVIDIA Hardware Discovery and Management Tool

A comprehensive command-line tool for discovering NVIDIA hardware and managing software updates across Unix variants in AI training and inference environments.

## Features

### 🔍 Hardware Discovery
- **GPU Detection**: Automatically discovers all NVIDIA GPUs in the system
- **Detailed Information**: Collects comprehensive GPU details including:
  - GPU name, driver version, CUDA version
  - Memory usage (total, used, free)
  - GPU and memory utilization
  - Temperature and power consumption
  - PCI bus ID, UUID, and compute capability

### 🛠️ Software Component Detection
- **NVIDIA Tools**: Detects installed NVIDIA tools and libraries
- **CUDA Toolkit**: Identifies CUDA installation and version
- **Python Packages**: Scans for NVIDIA-related Python packages
- **Container Support**: Checks Docker/Podman NVIDIA support
- **Version Grouping**: Groups components by version for easy analysis

### 🔄 Update Management
- **Update Checking**: Checks for available updates to NVIDIA components
- **Multiple Installation Methods**: Supports various installation methods:
  - APT (Ubuntu/Debian)
  - YUM/DNF (RHEL/CentOS/Fedora)
  - PIP (Python packages)
  - Conda (Conda packages)
  - Manual installation
- **Automated Updates**: Can automatically update drivers and software

### 📊 Comprehensive Reporting
- **Text Reports**: Human-readable detailed reports
- **JSON Output**: Machine-readable JSON format
- **Version Analysis**: Groups components by version
- **System Information**: Includes hostname, OS, kernel, and architecture

## Installation

### Quick Install
```bash
# Clone or download the tool
git clone <repository-url>
cd nvidia-discovery-tool

# Run the installation script
./install.sh
```

### Manual Install
```bash
# Make scripts executable
chmod +x nvidia_discovery.py update_manager.py

# Install Python dependencies (if needed)
pip3 install requests urllib3

# Test the installation
./nvidia_discovery.py --help
```

## Usage

### Basic Discovery
```bash
# Run complete discovery and display report
./nvidia_discovery.py

# Output in JSON format
./nvidia_discovery.py --json

# Save report to file
./nvidia_discovery.py --output report.txt

# Enable verbose logging
./nvidia_discovery.py --verbose
```

### Update Management
```bash
# Check for available updates
./nvidia_discovery.py --check-updates

# Update NVIDIA driver
./nvidia_discovery.py --update driver

# Update CUDA toolkit
./nvidia_discovery.py --update cuda

# Update with specific method
./nvidia_discovery.py --update driver --update-method apt
```

### Update Manager Standalone
```bash
# Check for updates
./update_manager.py --check

# Install/update component
./update_manager.py --install driver
./update_manager.py --install cuda --method apt
```

## Command Line Options

### Main Discovery Tool (`nvidia_discovery.py`)

| Option | Description |
|--------|-------------|
| `--json` | Output in JSON format |
| `--verbose`, `-v` | Enable verbose logging |
| `--output`, `-o` | Save output to file |
| `--check-updates` | Check for available updates |
| `--update` | Update specified component (driver, cuda, cudnn, tensorrt) |
| `--update-method` | Update method (auto, apt, yum, pip, conda, manual) |

### Update Manager (`update_manager.py`)

| Option | Description |
|--------|-------------|
| `--check` | Check for available updates |
| `--install` | Install/update specified component |
| `--method` | Installation method |
| `--verbose`, `-v` | Enable verbose logging |

## Supported Systems

### Operating Systems
- **Linux**: Ubuntu, Debian, RHEL, CentOS, Fedora, and other distributions
- **macOS**: macOS 10.14+ (limited NVIDIA support)
- **Unix variants**: Any system with Python 3.6+

### NVIDIA Components
- **Drivers**: All NVIDIA driver versions
- **CUDA**: CUDA Toolkit 8.0+
- **cuDNN**: cuDNN library detection
- **TensorRT**: TensorRT inference library
- **Container Runtime**: NVIDIA Container Runtime
- **Python Packages**: PyTorch, TensorFlow, CuPy, RAPIDS, etc.

## Example Output

### Text Report
```
================================================================================
NVIDIA HARDWARE AND SOFTWARE DISCOVERY REPORT
================================================================================
Generated: 2024-01-15 14:30:25 UTC

SYSTEM INFORMATION
----------------------------------------
Hostname: ai-training-server
OS: Linux 5.4.0-74-generic
Kernel: #83-Ubuntu SMP Sat May 8 02:35:39 UTC 2021
Architecture: x86_64
NVIDIA Driver: 535.154.05
CUDA Runtime: 12.3

GPU INFORMATION
----------------------------------------
Total GPUs: 4

GPU 0:
  Name: NVIDIA A100-SXM4-80GB
  Driver Version: 535.154.05
  CUDA Version: 12.3
  Memory: 1024/81920 MB
  Utilization: GPU 45%, Memory 12%
  Temperature: 65°C
  Power: 250/400 W
  UUID: GPU-12345678-1234-1234-1234-123456789abc
  PCI Bus ID: 00000000:00:1E.0
  Compute Capability: 8.0

SOFTWARE COMPONENTS
----------------------------------------

Version 535.154.05 (2 components):
  - nvidia-smi: Available
    Path: /usr/bin/nvidia-smi
  - nvidia-settings: Available
    Path: /usr/bin/nvidia-settings

Version 12.3 (3 components):
  - nvcc: Available
    Path: /usr/local/cuda/bin/nvcc
  - cuda: Available
    Path: /usr/local/cuda
  - python-cupy: Available
    Path: /usr/local/lib/python3.8/dist-packages/cupy
```

### JSON Output
```json
{
  "system_info": {
    "hostname": "ai-training-server",
    "os_name": "Linux",
    "os_version": "5.4.0-74-generic",
    "kernel_version": "#83-Ubuntu SMP Sat May 8 02:35:39 UTC 2021",
    "architecture": "x86_64",
    "nvidia_driver_version": "535.154.05",
    "cuda_runtime_version": "12.3"
  },
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA A100-SXM4-80GB",
      "driver_version": "535.154.05",
      "cuda_version": "12.3",
      "memory_total": "81920",
      "memory_used": "1024",
      "memory_free": "80896",
      "utilization_gpu": "45",
      "utilization_memory": "12",
      "temperature": "65",
      "power_draw": "250",
      "power_limit": "400",
      "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
      "pci_bus_id": "00000000:00:1E.0",
      "compute_capability": "8.0"
    }
  ],
  "software_components": [
    {
      "name": "nvidia-smi",
      "version": "535.154.05",
      "path": "/usr/bin/nvidia-smi",
      "status": "Available"
    }
  ],
  "discovery_timestamp": "2024-01-15 14:30:25 UTC"
}
```

## Requirements

### System Requirements
- Python 3.6 or higher
- NVIDIA drivers installed (for GPU detection)
- Internet connection (for update checking)

### Python Dependencies
- `requests` (for web requests)
- `urllib3` (for URL handling)

### Optional Dependencies
- `nvidia-ml-py3` (for enhanced GPU monitoring)
- `docker` (for container support detection)

## Troubleshooting

### Common Issues

1. **nvidia-smi not found**
   - Install NVIDIA drivers from [NVIDIA website](https://www.nvidia.com/drivers/)
   - Ensure drivers are properly loaded: `lsmod | grep nvidia`

2. **Permission denied errors**
   - Run with appropriate permissions for system-wide operations
   - Use `sudo` for driver updates

3. **Update checking fails**
   - Check internet connectivity
   - Verify NVIDIA website accessibility

4. **Python import errors**
   - Ensure Python 3.6+ is installed
   - Install required dependencies: `pip3 install requests urllib3`

### Debug Mode
```bash
# Enable verbose logging for debugging
./nvidia_discovery.py --verbose

# Check specific components
./nvidia_discovery.py --json | jq '.software_components[] | select(.name | contains("cuda"))'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include system information and error logs

## Changelog

### Version 1.0.0
- Initial release
- GPU discovery and reporting
- Software component detection
- Update management system
- Cross-platform support
- JSON and text output formats
