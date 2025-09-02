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
            cuda_runtime_version=cuda_runtime_version
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
            if len(parts) >= 5:  # Minimum required fields
                try:
                    # Handle different numbers of fields gracefully
                    gpu = GPUInfo(
                        index=int(parts[0]) if parts[0].isdigit() else 0,
                        name=parts[1] if len(parts) > 1 else "Unknown",
                        driver_version=parts[2] if len(parts) > 2 else "Unknown",
                        cuda_version=parts[3] if len(parts) > 3 else "Unknown",
                        memory_total=parts[4] if len(parts) > 4 else "Unknown",
                        memory_used=parts[5] if len(parts) > 5 else "Unknown",
                        memory_free=parts[6] if len(parts) > 6 else "Unknown",
                        utilization_gpu=parts[7] if len(parts) > 7 else "Unknown",
                        utilization_memory=parts[8] if len(parts) > 8 else "Unknown",
                        temperature=parts[9] if len(parts) > 9 else "Unknown",
                        power_draw=parts[10] if len(parts) > 10 else "Unknown",
                        power_limit=parts[11] if len(parts) > 11 else "Unknown",
                        uuid=parts[12] if len(parts) > 12 else "Unknown",
                        pci_bus_id=parts[13] if len(parts) > 13 else "Unknown",
                        compute_capability="Unknown"  # Will be filled later if available
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
        
        # Get comprehensive GPU information - try with basic fields first
        basic_query_fields = [
            'index', 'name', 'driver_version', 'cuda_version',
            'memory.total', 'memory.used', 'memory.free',
            'utilization.gpu', 'utilization.memory', 'temperature.gpu',
            'power.draw', 'power.limit', 'uuid', 'pci.bus_id'
        ]
        
        # Try basic query first
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
                return []
        
        gpus = self._parse_gpu_info(stdout)
        self.logger.info(f"Discovered {len(gpus)} GPU(s)")
        
        return gpus
    
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
        
        return components
    
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
        report.append(f"Kernel: {sys_info['kernel_version']}")
        report.append(f"Architecture: {sys_info['architecture']}")
        report.append(f"NVIDIA Driver: {sys_info['nvidia_driver_version']}")
        report.append(f"CUDA Runtime: {sys_info['cuda_runtime_version']}")
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

