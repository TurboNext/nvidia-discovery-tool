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
    package_managers: List[str]
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
        
        # Get CUDA runtime version
        success, stdout, _ = self._run_command(['nvcc', '--version'])
        cuda_runtime_version = "Unknown"
        if success and stdout:
            match = re.search(r'release (\d+\.\d+)', stdout)
            if match:
                cuda_runtime_version = match.group(1)
        
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
            
            # Try to get CUDA version from nvidia-smi version output
            if gpu.cuda_version == "Unknown":
                success, stdout, stderr = self._run_command(['nvidia-smi', '--version'])
                if success:
                    # Parse CUDA version from nvidia-smi version output
                    for line in stdout.split('\n'):
                        if 'CUDA Version:' in line:
                            cuda_ver = line.split('CUDA Version:')[1].strip().split()[0]
                            gpu.cuda_version = cuda_ver
                            break
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
        
        python_packages = [
            'pynvml',
            'nvidia-ml-py3',
            'cupy',
            'cudf',
            'rapids',
            'tensorflow-gpu',
            'torch',
            'tensorrt',
            'cudnn',
        ]
        
        for package in python_packages:
            version = self._get_python_package_version(package)
            path = self._find_python_package_path(package)
            status = "Available" if version != "Not found" else "Not found"
            
            components.append(SoftwareInfo(
                name=f"python-{package}",
                version=version,
                path=path,
                status=status
            ))
        
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
        
        # Check for previously missing tools
        missing_tools = self._check_missing_tools()
        components.extend(missing_tools)
        
        return components
    
    def _discover_additional_tools(self) -> List[SoftwareInfo]:
        """Discover additional NVIDIA tools and utilities"""
        components = []
        
        if self.verbose:
            self.logger.info("Checking for additional NVIDIA tools...")
        
        # Check for nvidia-settings
        success, stdout, stderr = self._run_command(['nvidia-settings', '--version'])
        if success:
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = version_match.group(1) if version_match else "Unknown"
            components.append(SoftwareInfo(
                name="nvidia-settings",
                version=version,
                path=self._find_tool_path('nvidia-settings'),
                status="Available"
            ))
        
        # Check for nvidia-xconfig
        success, stdout, stderr = self._run_command(['nvidia-xconfig', '--version'])
        if success:
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = version_match.group(1) if version_match else "Unknown"
            components.append(SoftwareInfo(
                name="nvidia-xconfig",
                version=version,
                path=self._find_tool_path('nvidia-xconfig'),
                status="Available"
            ))
        
        # Check for nvidia-modprobe
        success, stdout, stderr = self._run_command(['nvidia-modprobe', '--version'])
        if success:
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = version_match.group(1) if version_match else "Unknown"
            components.append(SoftwareInfo(
                name="nvidia-modprobe",
                version=version,
                path=self._find_tool_path('nvidia-modprobe'),
                status="Available"
            ))
        
        # Check for nvidia-persistenced
        success, stdout, stderr = self._run_command(['nvidia-persistenced', '--version'])
        if success:
            version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = version_match.group(1) if version_match else "Unknown"
            components.append(SoftwareInfo(
                name="nvidia-persistenced",
                version=version,
                path=self._find_tool_path('nvidia-persistenced'),
                status="Available"
            ))
        
        return components
    
    def _check_missing_tools(self) -> List[SoftwareInfo]:
        """Check for tools that were previously showing as 'Not found'"""
        components = []
        
        if self.verbose:
            self.logger.info("Checking for previously missing tools...")
        
        # List of tools to check with their expected commands
        tools_to_check = [
            ('nvidia-settings', ['nvidia-settings', '--version']),
            ('nvidia-xconfig', ['nvidia-xconfig', '--version']),
            ('nvidia-modprobe', ['nvidia-modprobe', '--version']),
            ('nvidia-persistenced', ['nvidia-persistenced', '--version']),
        ]
        
        for tool_name, cmd in tools_to_check:
            success, stdout, stderr = self._run_command(cmd)
            if success:
                version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                version = version_match.group(1) if version_match else "Unknown"
                components.append(SoftwareInfo(
                    name=tool_name,
                    version=version,
                    path=self._find_tool_path(tool_name),
                    status="Available"
                ))
                if self.verbose:
                    self.logger.info(f"Found {tool_name} version {version}")
            else:
                if self.verbose:
                    self.logger.debug(f"{tool_name} not found: {stderr}")
        
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
    
    def _discover_package_managers(self) -> List[str]:
        """Discover available package managers on the system"""
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
            success, _, _ = self._run_command(command.split())
            if success:
                package_managers.append(pm_name)
        
        return package_managers
    
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
    
    def collect_logs(self) -> Dict[str, Any]:
        """Collect system logs from multiple locations"""
        logs = {}
        
        if self.verbose:
            self.logger.info("Collecting system logs...")
        
        # Collect system logs from multiple locations
        system_logs = self._collect_system_logs()
        logs.update(system_logs)
        
        return logs
    
    def _collect_system_logs(self) -> Dict[str, Any]:
        """Collect system logs from various locations"""
        logs = {}
        
        # Define potential system log locations
        log_locations = [
            ('/var/log/syslog', 'syslog'),
            ('/var/log/kern.log', 'kernel'),
            ('/var/log/system.log', 'system'),
            ('/var/log/messages', 'messages'),
            ('/var/log/dmesg', 'dmesg')
        ]
        
        collected_logs = []
        
        for log_path, log_name in log_locations:
            try:
                if os.path.exists(log_path):
                    success, stdout, stderr = self._run_command(['tail', '-n', '50', log_path])
                    if success:
                        collected_logs.append({
                            'name': log_name,
                            'path': log_path,
                            'content': stdout,
                            'lines': len(stdout.split('\n'))
                        })
                        if self.verbose:
                            self.logger.info(f"Collected {log_name} log from {log_path}")
                    else:
                        if self.verbose:
                            self.logger.warning(f"Could not read {log_name} log: {stderr}")
                else:
                    if self.verbose:
                        self.logger.debug(f"{log_name} log not found at {log_path}")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Exception reading {log_name} log: {e}")
        
        
        logs['system_logs'] = collected_logs
        logs['system_log_count'] = len(collected_logs)
        
        return logs
    
    
    def run_discovery(self) -> Dict[str, Any]:
        """Run complete discovery process"""
        self.logger.info("Starting NVIDIA discovery process...")
        
        # Gather system information
        self.system_info = self._get_system_info()
        
        # Discover GPUs
        self.gpus = self.discover_gpus()
        
        # Discover software components
        self.software_components = self.discover_software_components()
        
        # Collect logs
        self.logs = self.collect_logs()
        
        return {
            'system_info': asdict(self.system_info),
            'gpus': [asdict(gpu) for gpu in self.gpus],
            'software_components': [asdict(comp) for comp in self.software_components],
            'logs': self.logs,
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
                report.append(f"  - {pm}")
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
                report.append(f"  CUDA Version: {gpu['cuda_version']}")
                report.append(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']} MB")
                report.append(f"  Utilization: GPU {gpu['utilization_gpu']}%, Memory {gpu['utilization_memory']}%")
                report.append(f"  Temperature: {gpu['temperature']}°C")
                report.append(f"  Power: {gpu['power_draw']}/{gpu['power_limit']} W")
                report.append(f"  UUID: {gpu['uuid']}")
                report.append(f"  PCI Bus ID: {gpu['pci_bus_id']}")
                report.append(f"  Compute Capability: {gpu['compute_capability']}")
                report.append("")
        
        # Software Components
        components = data['software_components']
        report.append("SOFTWARE COMPONENTS")
        report.append("-" * 40)
        
        # Group by version
        version_groups = defaultdict(list)
        for comp in components:
            version_groups[comp['version']].append(comp)
        
        for version, comps in sorted(version_groups.items()):
            report.append(f"\nVersion {version} ({len(comps)} components):")
            for comp in comps:
                report.append(f"  - {comp['name']}: {comp['status']}")
                if comp['path'] != "Not found":
                    report.append(f"    Path: {comp['path']}")
        
        # Log Information
        if 'logs' in data and data['logs']:
            report.append("LOG INFORMATION")
            report.append("-" * 40)
            
            logs = data['logs']
            
            # System logs
            if 'system_logs' in logs and logs['system_logs']:
                report.append(f"System Logs (found {logs.get('system_log_count', 0)} log files):")
                report.append("")
                
                for log_info in logs['system_logs']:
                    report.append(f"{log_info['name'].upper()} Log:")
                    report.append(f"  Path: {log_info['path']}")
                    report.append(f"  Lines: {log_info['lines']}")
                    report.append("")
                    
                    # Show first 10 lines of each log
                    log_lines = log_info['content'].split('\n')[:10]
                    for line in log_lines:
                        if line.strip():
                            report.append(f"    {line}")
                    if log_info['lines'] > 10:
                        report.append(f"    ... ({log_info['lines'] - 10} more lines)")
                    report.append("")
            else:
                report.append("System Logs: No system logs found")
                report.append("")
            
        
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

