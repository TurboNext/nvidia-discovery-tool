#!/usr/bin/env python3
"""
NVIDIA Hardware Discovery and Management Tool
A comprehensive tool for discovering NVIDIA hardware and managing software updates
across Unix variants in AI training and inference environments.
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    driver_version: str
    cuda_version: str
    memory_total: str
    memory_used: str
    memory_free: str
    utilization_gpu: str
    utilization_memory: str
    temperature: str
    power_draw: str
    power_limit: str
    uuid: str
    pci_bus_id: str
    compute_capability: str


@dataclass
class SoftwareInfo:
    """Information about NVIDIA software components"""
    name: str
    version: str
    path: str
    status: str


@dataclass
class SystemInfo:
    """System information"""
    hostname: str
    os_name: str
    os_version: str
    kernel_version: str
    architecture: str
    nvidia_driver_version: str
    cuda_runtime_version: str
    uname_output: str
    distribution: str
    package_managers: List[Dict[str, str]]
    disk_info: Dict[str, Any]
    network_info: str


class NVIDIADiscovery:
    """Main class for NVIDIA hardware and software discovery"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.system_info = None
        self.gpus = []
        self.software_components = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _run_command(self, command: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        """Run a shell command and return success status, stdout, and stderr"""
        try:
            self.logger.debug(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            return False, "", "Command timed out"
        except FileNotFoundError:
            self.logger.error(f"Command not found: {command[0]}")
            return False, "", f"Command not found: {command[0]}"
        except Exception as e:
            self.logger.error(f"Error running command: {e}")
            return False, "", str(e)
    
    def _get_system_info(self) -> SystemInfo:
        """Gather system information"""
        self.logger.info("Gathering system information...")
        
        hostname = platform.node()
        os_name = platform.system()
        os_version = platform.release()
        kernel_version = platform.version()
        architecture = platform.machine()
        
        # Get uname -a output
        success, uname_output, _ = self._run_command(['uname', '-a'])
        uname_output = uname_output.strip() if success else "Unknown"
        
        # Get distribution information
        distribution = self._get_distribution()
        
        # Get package managers
        package_managers = self._discover_package_managers()
        
        # Get disk information
        disk_info = self._get_disk_info()
        
        # Get network information
        success, network_info, _ = self._run_command(['ip', 'a'])
        network_info = network_info.strip() if success else "Unknown"
        
        # Get NVIDIA driver version
        success, stdout, _ = self._run_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'])
        nvidia_driver_version = stdout.strip().split('\n')[0] if success and stdout.strip() else "Unknown"
        
        # Get CUDA runtime version from nvidia-smi (primary source)
        cuda_runtime_version = "Unknown"
        success, stdout, _ = self._run_command(['nvidia-smi', '--version'])
        if success and stdout:
            # Always log nvidia-smi output for debugging
            self.logger.info(f"nvidia-smi --version output: {repr(stdout)}")
            
            # Try multiple patterns for CUDA version detection
            for line in stdout.split('\n'):
                line = line.strip()
                self.logger.info(f"Processing line: {repr(line)}")
                
                # Pattern 1: "CUDA Version: X.Y"
                if 'CUDA Version:' in line:
                    cuda_runtime_version = line.split('CUDA Version:')[1].strip().split()[0]
                    self.logger.info(f"Found CUDA version via 'CUDA Version:': {cuda_runtime_version}")
                    break
                
                # Pattern 2: "CUDA X.Y" (without "Version:")
                elif 'CUDA' in line and any(char.isdigit() for char in line):
                    import re
                    cuda_match = re.search(r'CUDA\s+(\d+\.\d+)', line)
                    if cuda_match:
                        cuda_runtime_version = cuda_match.group(1)
                        self.logger.info(f"Found CUDA version via regex: {cuda_runtime_version}")
                        break
                
                # Pattern 3: Look for any line with CUDA and version numbers
                elif 'CUDA' in line:
                    import re
                    version_match = re.search(r'(\d+\.\d+)', line)
                    if version_match:
                        cuda_runtime_version = version_match.group(1)
                        self.logger.info(f"Found CUDA version via general pattern: {cuda_runtime_version}")
                        break
            
            if cuda_runtime_version == "Unknown":
                self.logger.warning(f"Could not extract CUDA version from nvidia-smi output: {stdout}")
        
        return SystemInfo(
            hostname=hostname,
            os_name=os_name,
            os_version=os_version,
            kernel_version=kernel_version,
            architecture=architecture,
            nvidia_driver_version=nvidia_driver_version,
            cuda_runtime_version=cuda_runtime_version,
            uname_output=uname_output,
            distribution=distribution,
            package_managers=package_managers,
            disk_info=disk_info,
            network_info=network_info
        )
    
    def _parse_gpu_info(self, nvidia_smi_output: str) -> List[GPUInfo]:
        """Parse nvidia-smi output to extract GPU information"""
        gpus = []
        lines = nvidia_smi_output.strip().split('\n')
        
        if not lines or not lines[0].strip():
            return gpus
        
        # Process each line (no header in noheader format)
        for line in lines:
            if not line.strip():
                continue
                
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 13:  # We expect 13 fields from the working query
                try:
                    # Handle the 13-field output correctly
                    gpu = GPUInfo(
                        index=int(parts[0]) if parts[0].isdigit() else 0,
                        name=parts[1] if len(parts) > 1 else "Unknown",
                        driver_version=parts[2] if len(parts) > 2 else "Unknown",
                        cuda_version="Unknown",  # Not available in query
                        memory_total=parts[3] if len(parts) > 3 else "Unknown",
                        memory_used=parts[4] if len(parts) > 4 else "Unknown",
                        memory_free=parts[5] if len(parts) > 5 else "Unknown",
                        utilization_gpu=parts[6] if len(parts) > 6 else "Unknown",
                        utilization_memory=parts[7] if len(parts) > 7 else "Unknown",
                        temperature=parts[8] if len(parts) > 8 else "Unknown",
                        power_draw=parts[9] if len(parts) > 9 else "Unknown",
                        power_limit=parts[10] if len(parts) > 10 else "Unknown",
                        uuid=parts[11] if len(parts) > 11 else "Unknown",
                        pci_bus_id=parts[12] if len(parts) > 12 else "Unknown",
                        compute_capability="Unknown"  # Not available in query
                    )
                    gpus.append(gpu)
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing GPU info line: {line} - {e}")
        
        return gpus
    
    def discover_gpus(self) -> List[GPUInfo]:
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
        
        # Try to get additional information for each GPU
        for gpu in gpus:
            self._enrich_gpu_info(gpu)
        
        self.logger.info(f"Discovered {len(gpus)} GPU(s)")
        
        return gpus
    
    def _enrich_gpu_info(self, gpu: GPUInfo) -> None:
        """Try to get additional GPU information using alternative methods"""
        try:
            # Try to get compute capability using nvidia-smi with individual GPU query
            success, stdout, stderr = self._run_command([
                'nvidia-smi', f'--id={gpu.index}',
                '--query-gpu=compute_cap', '--format=csv,noheader,nounits'
            ])
            if success and stdout.strip():
                gpu.compute_capability = stdout.strip()
            
        except Exception as e:
            self.logger.debug(f"Could not enrich GPU {gpu.index} info: {e}")
    
    def discover_software_components(self) -> List[SoftwareInfo]:
        """Discover NVIDIA software components and their versions"""
        self.logger.info("Discovering NVIDIA software components...")
        
        components = []
        
        # List of NVIDIA tools to check
        nvidia_tools = [
            ('nvidia-smi', 'NVIDIA System Management Interface'),
            ('nvcc', 'NVIDIA CUDA Compiler'),
            ('nvidia-ml-py', 'NVIDIA Management Library Python bindings'),
            ('cuda', 'CUDA Toolkit'),
            ('cudnn', 'cuDNN Library'),
            ('tensorrt', 'TensorRT'),
            ('nvidia-container-runtime', 'NVIDIA Container Runtime'),
            ('nvidia-docker', 'NVIDIA Docker'),
            ('nvidia-settings', 'NVIDIA Settings'),
            ('nvidia-xconfig', 'NVIDIA X Configuration'),
            ('nvidia-modprobe', 'NVIDIA Module Probe'),
            ('nvidia-persistenced', 'NVIDIA Persistence Daemon'),
            ('nvidia-ctk', 'NVIDIA Container Toolkit'),
        ]
        
        for tool, description in nvidia_tools:
            version = self._get_tool_version(tool)
            path = self._find_tool_path(tool)
            status = "Available" if version != "Not found" else "Not found"
            
            components.append(SoftwareInfo(
                name=tool,
                version=version,
                path=path,
                status=status
            ))
        
        # Check for additional CUDA libraries
        cuda_libs = self._discover_cuda_libraries()
        components.extend(cuda_libs)
        
        # Check for Python packages
        python_packages = self._discover_python_packages()
        components.extend(python_packages)
        
        # Check for container runtime support
        container_support = self._discover_container_support()
        components.extend(container_support)
        
        return components
    
    def _get_tool_version(self, tool: str) -> str:
        """Get version of a specific tool"""
        version_commands = {
            'nvidia-smi': ['nvidia-smi', '--version'],
            'nvcc': ['nvcc', '--version'],
            'nvidia-ml-py': ['python3', '-c', 'import pynvml; print(pynvml.nvmlSystemGetDriverVersion())'],
            'cuda': ['nvcc', '--version'],
            'cudnn': ['python3', '-c', 'import cudnn; print(cudnn.version())'],
            'tensorrt': ['python3', '-c', 'import tensorrt; print(tensorrt.__version__)'],
            'nvidia-container-runtime': ['nvidia-container-runtime', '--version'],
            'nvidia-docker': ['nvidia-docker', '--version'],
        }
        
        if tool in version_commands:
            success, stdout, _ = self._run_command(version_commands[tool])
            if success and stdout.strip():
                # Extract version number from output
                version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                if version_match:
                    return version_match.group(1)
                return stdout.strip().split('\n')[0]
        
        return "Not found"
    
    def _find_tool_path(self, tool: str) -> str:
        """Find the path to a tool"""
        success, stdout, _ = self._run_command(['which', tool])
        if success and stdout.strip():
            return stdout.strip()
        return "Not found"
    
    def _discover_cuda_libraries(self) -> List[SoftwareInfo]:
        """Discover CUDA libraries in common locations"""
        components = []
        
        # Common CUDA library paths
        cuda_paths = [
            '/usr/local/cuda',
            '/opt/cuda',
            '/usr/cuda',
            '/usr/local/cuda-*',
        ]
        
        for base_path in cuda_paths:
            if '*' in base_path:
                # Handle glob patterns
                import glob
                paths = glob.glob(base_path)
            else:
                paths = [base_path] if os.path.exists(base_path) else []
            
            for path in paths:
                if os.path.isdir(path):
                    # Check for various CUDA components
                    lib_path = os.path.join(path, 'lib64')
                    if os.path.exists(lib_path):
                        # Look for specific libraries
                        for lib_name in ['libcudnn.so', 'libcublas.so', 'libcurand.so']:
                            lib_file = os.path.join(lib_path, lib_name)
                            if os.path.exists(lib_file):
                                version = self._get_library_version(lib_file)
                                components.append(SoftwareInfo(
                                    name=f"{lib_name} ({path})",
                                    version=version,
                                    path=lib_file,
                                    status="Available"
                                ))
        
        return components
    
    def _get_library_version(self, lib_path: str) -> str:
        """Get version of a library file"""
        try:
            success, stdout, _ = self._run_command(['ldd', lib_path])
            if success:
                # Try to extract version from ldd output
                version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                if version_match:
                    return version_match.group(1)
        except:
            pass
        return "Unknown"
    
    def _discover_python_packages(self) -> List[SoftwareInfo]:
        """Discover NVIDIA-related Python packages"""
        components = []
        
        # Removed python package discovery as requested
        # python_packages = [
        #     'pynvml',
        #     'nvidia-ml-py3',
        #     'cupy',
        #     'cudf',
        #     'rapids',
        #     'tensorflow-gpu',
        #     'torch',
        #     'tensorrt',
        #     'cudnn',
        # ]
        
        return components
    
    def _get_python_package_version(self, package: str) -> str:
        """Get version of a Python package"""
        try:
            success, stdout, _ = self._run_command(['python3', '-c', f'import {package}; print({package}.__version__)'])
            if success and stdout.strip():
                return stdout.strip()
        except:
            pass
        
        # Try pip show as fallback
        success, stdout, _ = self._run_command(['pip3', 'show', package])
        if success and stdout:
            for line in stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        
        return "Not found"
    
    def _find_python_package_path(self, package: str) -> str:
        """Find the path to a Python package"""
        try:
            success, stdout, _ = self._run_command(['python3', '-c', f'import {package}; print({package}.__file__)'])
            if success and stdout.strip():
                return stdout.strip()
        except:
            pass
        return "Not found"
    
    def _discover_container_support(self) -> List[SoftwareInfo]:
        """Discover container runtime support for NVIDIA"""
        components = []
        
        # Check for Docker
        success, stdout, _ = self._run_command(['docker', '--version'])
        if success:
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = version_match.group(1) if version_match else "Unknown"
            components.append(SoftwareInfo(
                name="docker",
                version=version,
                path=self._find_tool_path('docker'),
                status="Available"
            ))
        
        # Check for NVIDIA Docker support
        success, stdout, _ = self._run_command(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi'])
        nvidia_docker_support = "Available" if success else "Not available"
        components.append(SoftwareInfo(
            name="nvidia-docker-support",
            version="N/A",
            path="Docker runtime",
            status=nvidia_docker_support
        ))
        
        # Check for Podman
        success, stdout, _ = self._run_command(['podman', '--version'])
        if success:
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = version_match.group(1) if version_match else "Unknown"
            components.append(SoftwareInfo(
                name="podman",
                version=version,
                path=self._find_tool_path('podman'),
                status="Available"
            ))
        
        # Discover additional NVIDIA tools
        additional_components = self._discover_additional_tools()
        components.extend(additional_components)
        
        
        # Discover NVIDIA Container Toolkit configuration
        nvidia_ctk_config = self._discover_nvidia_ctk_config()
        components.extend(nvidia_ctk_config)
        
        return components
    
    def _discover_additional_tools(self) -> List[SoftwareInfo]:
        """Discover additional NVIDIA tools and utilities"""
        components = []
        
        if self.verbose:
            self.logger.info("Checking for additional NVIDIA tools...")
        
        # Additional tools that are not in the main nvidia_tools list
        additional_tools = [
            'nvidia-ctk',
            'nvidia-container-cli',
            'nvidia-cuda-mps-server',
            'nvidia-cuda-mps-control'
        ]
        
        for tool in additional_tools:
            success, stdout, stderr = self._run_command([tool, '--version'])
            if success:
                version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                version = version_match.group(1) if version_match else "Unknown"
                components.append(SoftwareInfo(
                    name=tool,
                    version=version,
                    path=self._find_tool_path(tool),
                    status="Available"
                ))
        
        return components
    
    
    def _discover_nvidia_ctk_config(self) -> List[SoftwareInfo]:
        """Discover NVIDIA Container Toolkit configuration"""
        components = []
        
        if self.verbose:
            self.logger.info("Checking for NVIDIA Container Toolkit configuration...")
        
        # Check for nvidia-ctk binary
        success, stdout, _ = self._run_command(['which', 'nvidia-ctk'])
        if success and stdout.strip():
            nvidia_ctk_path = stdout.strip()
            
            # Get version
            success, version_output, _ = self._run_command(['nvidia-ctk', '--version'])
            version = "Unknown"
            if success and version_output:
                version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', version_output)
                if version_match:
                    version = version_match.group(1)
            
            components.append(SoftwareInfo(
                name="nvidia-ctk",
                version=version,
                path=nvidia_ctk_path,
                status="Available"
            ))
            
            # Check nvidia-ctk configuration
            config_paths = [
                '/etc/nvidia-container-runtime/config.toml',
                '/usr/local/nvidia-toolkit/config.toml',
                '/etc/nvidia-container-toolkit/config.toml'
            ]
            
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config_content = f.read()
                            components.append(SoftwareInfo(
                                name=f"nvidia-ctk-config ({config_path})",
                                version="Config",
                                path=config_path,
                                status="Available"
                            ))
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(f"Could not read nvidia-ctk config {config_path}: {e}")
        
        # Check for nvidia-container-cli
        success, stdout, _ = self._run_command(['which', 'nvidia-container-cli'])
        if success and stdout.strip():
            components.append(SoftwareInfo(
                name="nvidia-container-cli",
                version="Available",
                path=stdout.strip(),
                status="Available"
            ))
        
        return components
    
    def _get_distribution(self) -> str:
        """Get Linux distribution information"""
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=')[1].strip('"')
                    elif line.startswith('NAME='):
                        return line.split('=')[1].strip('"')
        except:
            pass
        return "Unknown"
    
    def _discover_package_managers(self) -> List[Dict[str, str]]:
        """Discover available package managers on the system with versions"""
        package_managers = []
        
        # List of Unix/Linux package managers to check
        pm_commands = [
            # Debian/Ubuntu family
            ('apt', 'apt --version'),
            ('apt-get', 'apt-get --version'),
            ('dpkg', 'dpkg --version'),
            ('aptitude', 'aptitude --version'),
            
            # Red Hat/CentOS/Fedora family
            ('yum', 'yum --version'),
            ('dnf', 'dnf --version'),
            ('rpm', 'rpm --version'),
            
            # SUSE family
            ('zypper', 'zypper --version'),
            
            # Arch Linux
            ('pacman', 'pacman --version'),
            
            # Gentoo
            ('portage', 'emerge --version'),
            
            # Universal package managers
            ('snap', 'snap --version'),
            ('flatpak', 'flatpak --version'),
            ('appimage', 'appimage --version'),
            
            # Language-specific package managers
            ('pip', 'pip --version'),
            ('pip3', 'pip3 --version'),
            ('conda', 'conda --version'),
            ('npm', 'npm --version'),
            ('yarn', 'yarn --version'),
            ('gem', 'gem --version'),
            ('cargo', 'cargo --version'),
            ('go', 'go version'),
            
            # Container package managers
            ('docker', 'docker --version'),
            ('podman', 'podman --version'),
        ]
        
        for pm_name, command in pm_commands:
            success, stdout, _ = self._run_command(command.split())
            if success:
                version = self._extract_version_from_output(stdout, pm_name)
                package_managers.append({
                    'name': pm_name,
                    'version': version
                })
        
        return package_managers
    
    def _extract_version_from_output(self, output: str, pm_name: str) -> str:
        """Extract version from package manager output"""
        if not output:
            return "Unknown"
        
        # Try to find version pattern
        version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', output)
        if version_match:
            return version_match.group(1)
        
        # Return first line if no version pattern found
        return output.split('\n')[0].strip()
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get system disk information"""
        disk_info = {}
        
        # Get disk usage using df command
        success, stdout, _ = self._run_command(['df', '-h'])
        if success:
            disk_info['df_output'] = stdout.strip()
        
        # Get disk devices using lsblk
        success, stdout, _ = self._run_command(['lsblk'])
        if success:
            disk_info['lsblk_output'] = stdout.strip()
        
        # Get disk space summary
        success, stdout, _ = self._run_command(['df', '-h', '/'])
        if success:
            lines = stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    disk_info['root_filesystem'] = {
                        'device': parts[0],
                        'size': parts[1],
                        'used': parts[2],
                        'available': parts[3],
                        'use_percent': parts[4] if len(parts) > 4 else 'Unknown',
                        'mounted_on': parts[5] if len(parts) > 5 else 'Unknown'
                    }
        
        return disk_info
    
    def _load_thresholds(self) -> List[Dict[str, str]]:
        """Load thresholds from CSV file"""
        thresholds = []
        csv_path = os.path.join(os.path.dirname(__file__), 'thresholds.csv')
        
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            thresholds.append({
                                'component': parts[0],
                                'threshold': parts[1],
                                'operator': parts[2],
                                'description': parts[3] if len(parts) > 3 else ''
                            })
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Could not load thresholds: {e}")
        
        return thresholds
    
    def _check_thresholds(self, data: Dict[str, Any], thresholds: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Check measured values against thresholds"""
        results = []
        
        if self.verbose:
            self.logger.info(f"Checking {len(thresholds)} thresholds")
        
        for threshold in thresholds:
            component = threshold['component']
            threshold_value = threshold['threshold']
            operator = threshold['operator']
            
            # Get measured value based on component
            measured_value = self._get_measured_value(data, component)
            
            if self.verbose:
                self.logger.info(f"Threshold check: {component} = {measured_value} vs {threshold_value} ({operator})")
            
            if measured_value:
                passed = self._compare_versions(measured_value, threshold_value, operator)
                results.append({
                    'component': component,
                    'measured': measured_value,
                    'threshold': threshold_value,
                    'passed': passed
                })
                if self.verbose:
                    self.logger.info(f"Threshold result: {component} = {passed}")
        
        if self.verbose:
            self.logger.info(f"Threshold results: {len(results)} checks completed")
        
        return results
    
    def _get_measured_value(self, data: Dict[str, Any], component: str) -> str:
        """Get measured value for a component"""
        if component == 'cuda_version':
            # Get CUDA version from GPU info first, then system info
            gpus = data.get('gpus', [])
            if gpus and len(gpus) > 0:
                cuda_version = gpus[0].get('cuda_version', 'Unknown')
                if cuda_version != 'Unknown':
                    return cuda_version
            
            # Fallback to system info
            system_cuda = data.get('system_info', {}).get('cuda_runtime_version', 'Unknown')
            if system_cuda != 'Unknown':
                return system_cuda
            
            # Last resort: try to get from nvidia-smi directly
            success, stdout, _ = self._run_command(['nvidia-smi', '--version'])
            if success and stdout:
                # Simple and reliable CUDA version detection from nvidia-smi
                for line in stdout.split('\n'):
                    line = line.strip()
                    # Look for "CUDA Version: X.Y" pattern
                    if 'CUDA Version:' in line:
                        return line.split('CUDA Version:')[1].strip().split()[0]
            
            return 'Unknown'
        
        elif component == 'cudnn':
            # Get cuDNN version from software components
            components = data.get('software_components', [])
            for comp in components:
                if comp.get('name') == 'cudnn':
                    return comp.get('version', 'Unknown')
            return 'Unknown'
        
        return "Unknown"
    
    def _compare_versions(self, measured: str, threshold: str, operator: str) -> bool:
        """Compare version strings"""
        # If measured value is Unknown, fail the threshold
        if measured == "Unknown" or measured == "Not found":
            return False
            
        try:
            # Convert version strings to comparable format
            measured_parts = [int(x) for x in measured.split('.')]
            threshold_parts = [int(x) for x in threshold.split('.')]
            
            # Pad with zeros if needed
            max_len = max(len(measured_parts), len(threshold_parts))
            measured_parts.extend([0] * (max_len - len(measured_parts)))
            threshold_parts.extend([0] * (max_len - len(threshold_parts)))
            
            if operator == '>=':
                return measured_parts >= threshold_parts
            elif operator == '>':
                return measured_parts > threshold_parts
            elif operator == '<=':
                return measured_parts <= threshold_parts
            elif operator == '<':
                return measured_parts < threshold_parts
            elif operator == '==':
                return measured_parts == threshold_parts
            else:
                return False
        except:
            return False
    
    def run_discovery(self) -> Dict[str, Any]:
        """Run complete discovery process"""
        self.logger.info("Starting NVIDIA discovery process...")
        
        # Gather system information
        self.system_info = self._get_system_info()
        
        # Discover GPUs
        self.gpus = self.discover_gpus()
        
        # Discover software components
        self.software_components = self.discover_software_components()
        
        return {
            'system_info': asdict(self.system_info),
            'gpus': [asdict(gpu) for gpu in self.gpus],
            'software_components': [asdict(comp) for comp in self.software_components],
            'discovery_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
    
    def generate_report(self, data: Dict[str, Any], format: str = 'text') -> str:
        """Generate a comprehensive report"""
        if format == 'json':
            return json.dumps(data, indent=2)
        
        # Generate text report
        report = []
        report.append("=" * 80)
        report.append("NVIDIA HARDWARE AND SOFTWARE DISCOVERY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {data['discovery_timestamp']}")
        report.append("")
        
        # System Information
        sys_info = data['system_info']
        report.append("SYSTEM INFORMATION")
        report.append("-" * 40)
        report.append(f"Hostname: {sys_info['hostname']}")
        report.append(f"OS: {sys_info['os_name']} {sys_info['os_version']}")
        report.append(f"Distribution: {sys_info['distribution']}")
        report.append(f"Kernel: {sys_info['kernel_version']}")
        report.append(f"Architecture: {sys_info['architecture']}")
        report.append(f"NVIDIA Driver: {sys_info['nvidia_driver_version']}")
        report.append(f"CUDA Runtime: {sys_info['cuda_runtime_version']}")
        report.append("")
        
        # Uname output
        report.append("SYSTEM DETAILS")
        report.append("-" * 40)
        report.append(f"Uname: {sys_info['uname_output']}")
        report.append("")
        
        # Package Managers
        if sys_info['package_managers']:
            report.append("PACKAGE MANAGERS")
            report.append("-" * 40)
            for pm in sys_info['package_managers']:
                report.append(f"  - {pm['name']}: {pm['version']}")
            report.append("")
        
        # Disk Information
        if 'disk_info' in sys_info and sys_info['disk_info']:
            report.append("DISK INFORMATION")
            report.append("-" * 40)
            
            if 'root_filesystem' in sys_info['disk_info']:
                root_fs = sys_info['disk_info']['root_filesystem']
                report.append(f"Root Filesystem: {root_fs['device']}")
                report.append(f"  Size: {root_fs['size']}")
                report.append(f"  Used: {root_fs['used']}")
                report.append(f"  Available: {root_fs['available']}")
                report.append(f"  Use%: {root_fs['use_percent']}")
                report.append(f"  Mounted on: {root_fs['mounted_on']}")
                report.append("")
            
            if 'df_output' in sys_info['disk_info']:
                report.append("Disk Usage (df -h):")
                df_lines = sys_info['disk_info']['df_output'].split('\n')[:10]  # Show first 10 lines
                for line in df_lines:
                    report.append(f"  {line}")
                if len(sys_info['disk_info']['df_output'].split('\n')) > 10:
                    report.append(f"  ... ({len(sys_info['disk_info']['df_output'].split('\n')) - 10} more lines)")
                report.append("")
        
        # Network Information
        if 'network_info' in sys_info and sys_info['network_info'] != "Unknown":
            report.append("NETWORK INFORMATION")
            report.append("-" * 40)
            network_lines = sys_info['network_info'].split('\n')[:20]  # Show first 20 lines
            for line in network_lines:
                if line.strip():
                    report.append(f"  {line}")
            if len(sys_info['network_info'].split('\n')) > 20:
                report.append(f"  ... ({len(sys_info['network_info'].split('\n')) - 20} more lines)")
            report.append("")
        
        # GPU Information
        gpus = data['gpus']
        report.append("GPU INFORMATION")
        report.append("-" * 40)
        report.append(f"Total GPUs: {len(gpus)}")
        report.append("")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                report.append(f"GPU {i}:")
                report.append(f"  Name: {gpu['name']}")
                report.append(f"  Driver Version: {gpu['driver_version']}")
                report.append(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']} MB")
                report.append(f"  Utilization: GPU {gpu['utilization_gpu']}%, Memory {gpu['utilization_memory']}%")
                report.append(f"  Temperature: {gpu['temperature']}°C")
                report.append(f"  Power: {gpu['power_draw']}/{gpu['power_limit']} W")
                report.append(f"  UUID: {gpu['uuid']}")
                report.append(f"  PCI Bus ID: {gpu['pci_bus_id']}")
                report.append(f"  Compute Capability: {gpu['compute_capability']}")
                report.append("")
        
        # GPU Summary
        if gpus:
            report.append("GPU SUMMARY")
            report.append("-" * 40)
            
            # Count unique Name/Driver Version combinations
            gpu_combinations = {}
            for gpu in gpus:
                key = f"{gpu['name']} (Driver: {gpu['driver_version']})"
                gpu_combinations[key] = gpu_combinations.get(key, 0) + 1
            
            # Display summary
            for combination, count in sorted(gpu_combinations.items()):
                report.append(f"  {combination}: {count} GPU(s)")
            report.append("")
        
        # Software Components
        components = data['software_components']
        report.append("SOFTWARE COMPONENTS")
        report.append("-" * 40)
        
        # Table header
        report.append("Component,Version,Status,Path")
        
        # List each component in table format
        for comp in sorted(components, key=lambda x: x['name']):
            path = comp['path'] if comp['path'] != "Not found" else "N/A"
            report.append(f"{comp['name']},{comp['version']},{comp['status']},{path}")
        
        report.append("")
        
        # Threshold Information
        thresholds = self._load_thresholds()
        threshold_results = self._check_thresholds(data, thresholds)
        
        if threshold_results:
            report.append("THRESHOLD REPORT")
            report.append("-" * 40)
            report.append(f"{'Component':<20} {'Measured':<15} {'Threshold':<15} {'Status':<10}")
            report.append("-" * 60)
            for result in threshold_results:
                status = "PASS" if result['passed'] else "FAIL"
                report.append(f"{result['component']:<20} {result['measured']:<15} {result['threshold']:<15} {status:<10}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="NVIDIA Hardware Discovery and Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run discovery and display report
  %(prog)s --json             # Output in JSON format
  %(prog)s --verbose          # Enable verbose logging
  %(prog)s --output report.txt # Save report to file
  %(prog)s --check-updates    # Check for available updates
  %(prog)s --update driver    # Update NVIDIA driver
  %(prog)s --update cuda      # Update CUDA toolkit
        """
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    parser.add_argument(
        '--check-updates',
        action='store_true',
        help='Check for available updates'
    )
    
    parser.add_argument(
        '--update',
        type=str,
        choices=['driver', 'cuda', 'cudnn', 'tensorrt'],
        help='Update specified component'
    )
    
    parser.add_argument(
        '--update-method',
        type=str,
        default='auto',
        choices=['auto', 'apt', 'yum', 'pip', 'conda', 'manual'],
        help='Update method to use'
    )
    
    args = parser.parse_args()
    
    # Create discovery instance
    discovery = NVIDIADiscovery(verbose=args.verbose)
    
    try:
        if args.check_updates or args.update:
            # Import update manager
            from update_manager import NVIDIAUpdateManager
            update_manager = NVIDIAUpdateManager(verbose=args.verbose)
            
            if args.check_updates:
                # Check for updates
                updates = update_manager.check_updates()
                report = update_manager.generate_update_report(updates)
                print(report)
            
            elif args.update:
                # Update component
                success = update_manager.install_component(args.update, args.update_method)
                if success:
                    print(f"Successfully updated {args.update}")
                else:
                    print(f"Failed to update {args.update}")
                    sys.exit(1)
        
        else:
            # Run discovery
            data = discovery.run_discovery()
            
            # Generate report
            format_type = 'json' if args.json else 'text'
            report = discovery.generate_report(data, format_type)
            
            # Output report
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"Report saved to: {args.output}")
            else:
                print(report)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during operation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

