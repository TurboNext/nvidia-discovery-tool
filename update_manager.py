#!/usr/bin/env python3
"""
NVIDIA Update Manager
Handles checking for and implementing updates to NVIDIA tools and libraries.
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re


@dataclass
class UpdateInfo:
    """Information about available updates"""
    component: str
    current_version: str
    latest_version: str
    update_available: bool
    download_url: str
    release_notes: str
    compatibility: str


@dataclass
class InstallationMethod:
    """Information about installation method"""
    method: str  # 'apt', 'yum', 'dnf', 'pip', 'conda', 'manual'
    command: str
    description: str


class NVIDIAUpdateManager:
    """Manages updates for NVIDIA software components"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.system_info = self._get_system_info()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for compatibility checking"""
        return {
            'os_name': platform.system(),
            'os_version': platform.release(),
            'architecture': platform.machine(),
            'distribution': self._get_distribution()
        }
    
    def _get_distribution(self) -> str:
        """Get Linux distribution information"""
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('ID='):
                        return line.split('=')[1].strip('"')
        except:
            pass
        return "unknown"
    
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
    
    def _get_latest_driver_version(self) -> Optional[str]:
        """Get the latest NVIDIA driver version from NVIDIA's website"""
        try:
            # This is a simplified approach - in practice, you'd parse NVIDIA's driver page
            # For now, we'll use a mock approach
            url = "https://www.nvidia.com/drivers/latest-driver-version/"
            
            # In a real implementation, you would:
            # 1. Fetch the NVIDIA driver download page
            # 2. Parse the HTML to extract the latest version
            # 3. Return the version string
            
            # For demonstration, return a mock version
            return "535.154.05"
            
        except Exception as e:
            self.logger.error(f"Failed to get latest driver version: {e}")
            return None
    
    def _get_latest_cuda_version(self) -> Optional[str]:
        """Get the latest CUDA version"""
        try:
            # Similar to driver version, this would fetch from NVIDIA's CUDA page
            # For demonstration, return a mock version
            return "12.3"
        except Exception as e:
            self.logger.error(f"Failed to get latest CUDA version: {e}")
            return None
    
    def _get_current_driver_version(self) -> str:
        """Get current NVIDIA driver version"""
        success, stdout, _ = self._run_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'])
        if success and stdout.strip():
            return stdout.strip().split('\n')[0]
        return "Unknown"
    
    def _get_current_cuda_version(self) -> str:
        """Get current CUDA version"""
        success, stdout, _ = self._run_command(['nvcc', '--version'])
        if success and stdout:
            match = re.search(r'release (\d+\.\d+)', stdout)
            if match:
                return match.group(1)
        return "Unknown"
    
    def _compare_versions(self, current: str, latest: str) -> bool:
        """Compare version strings to determine if update is available"""
        if current == "Unknown" or latest is None:
            return False
        
        try:
            # Simple version comparison - in practice, use proper version parsing
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Pad with zeros if needed
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            return current_parts < latest_parts
        except:
            return False
    
    def check_updates(self, components: List[str] = None) -> List[UpdateInfo]:
        """Check for updates for specified components"""
        if components is None:
            components = ['driver', 'cuda', 'cudnn', 'tensorrt']
        
        updates = []
        
        for component in components:
            self.logger.info(f"Checking updates for {component}...")
            
            if component == 'driver':
                current = self._get_current_driver_version()
                latest = self._get_latest_driver_version()
                update_available = self._compare_versions(current, latest)
                
                updates.append(UpdateInfo(
                    component='NVIDIA Driver',
                    current_version=current,
                    latest_version=latest or "Unknown",
                    update_available=update_available,
                    download_url="https://www.nvidia.com/drivers/",
                    release_notes="See NVIDIA driver release notes",
                    compatibility=f"Compatible with {self.system_info['os_name']} {self.system_info['architecture']}"
                ))
            
            elif component == 'cuda':
                current = self._get_current_cuda_version()
                latest = self._get_latest_cuda_version()
                update_available = self._compare_versions(current, latest)
                
                updates.append(UpdateInfo(
                    component='CUDA Toolkit',
                    current_version=current,
                    latest_version=latest or "Unknown",
                    update_available=update_available,
                    download_url="https://developer.nvidia.com/cuda-downloads",
                    release_notes="See CUDA release notes",
                    compatibility=f"Compatible with {self.system_info['os_name']} {self.system_info['architecture']}"
                ))
            
            # Add more components as needed
        
        return updates
    
    def get_installation_methods(self, component: str) -> List[InstallationMethod]:
        """Get available installation methods for a component"""
        methods = []
        
        if component.lower() == 'driver':
            if self.system_info['os_name'] == 'Linux':
                if self.system_info['distribution'] in ['ubuntu', 'debian']:
                    methods.append(InstallationMethod(
                        method='apt',
                        command='sudo apt update && sudo apt install nvidia-driver-535',
                        description='Install via APT package manager'
                    ))
                elif self.system_info['distribution'] in ['rhel', 'centos', 'fedora']:
                    methods.append(InstallationMethod(
                        method='yum',
                        command='sudo yum install nvidia-driver',
                        description='Install via YUM package manager'
                    ))
                
                methods.append(InstallationMethod(
                    method='manual',
                    command='Download from NVIDIA website and run installer',
                    description='Manual installation from NVIDIA website'
                ))
        
        elif component.lower() == 'cuda':
            if self.system_info['os_name'] == 'Linux':
                if self.system_info['distribution'] in ['ubuntu', 'debian']:
                    methods.append(InstallationMethod(
                        method='apt',
                        command='sudo apt install cuda-toolkit-12-3',
                        description='Install CUDA via APT'
                    ))
                elif self.system_info['distribution'] in ['rhel', 'centos', 'fedora']:
                    methods.append(InstallationMethod(
                        method='yum',
                        command='sudo yum install cuda-toolkit',
                        description='Install CUDA via YUM'
                    ))
            
            methods.append(InstallationMethod(
                method='manual',
                command='Download CUDA installer from NVIDIA website',
                description='Manual CUDA installation'
            ))
        
        return methods
    
    def install_component(self, component: str, method: str = 'auto') -> bool:
        """Install or update a component using the specified method"""
        self.logger.info(f"Installing/updating {component} using method: {method}")
        
        if method == 'auto':
            methods = self.get_installation_methods(component)
            if not methods:
                self.logger.error(f"No installation methods available for {component}")
                return False
            method = methods[0].method
        
        if method == 'apt':
            return self._install_via_apt(component)
        elif method == 'yum':
            return self._install_via_yum(component)
        elif method == 'pip':
            return self._install_via_pip(component)
        elif method == 'conda':
            return self._install_via_conda(component)
        elif method == 'manual':
            return self._install_manual(component)
        else:
            self.logger.error(f"Unknown installation method: {method}")
            return False
    
    def _install_via_apt(self, component: str) -> bool:
        """Install component via APT"""
        commands = {
            'driver': ['sudo', 'apt', 'update', '&&', 'sudo', 'apt', 'install', '-y', 'nvidia-driver-535'],
            'cuda': ['sudo', 'apt', 'install', '-y', 'cuda-toolkit-12-3'],
            'cudnn': ['sudo', 'apt', 'install', '-y', 'libcudnn8-dev'],
        }
        
        if component not in commands:
            self.logger.error(f"APT installation not supported for {component}")
            return False
        
        success, stdout, stderr = self._run_command(commands[component])
        if success:
            self.logger.info(f"Successfully installed {component} via APT")
            return True
        else:
            self.logger.error(f"Failed to install {component} via APT: {stderr}")
            return False
    
    def _install_via_yum(self, component: str) -> bool:
        """Install component via YUM/DNF"""
        commands = {
            'driver': ['sudo', 'yum', 'install', '-y', 'nvidia-driver'],
            'cuda': ['sudo', 'yum', 'install', '-y', 'cuda-toolkit'],
            'cudnn': ['sudo', 'yum', 'install', '-y', 'libcudnn8-devel'],
        }
        
        if component not in commands:
            self.logger.error(f"YUM installation not supported for {component}")
            return False
        
        success, stdout, stderr = self._run_command(commands[component])
        if success:
            self.logger.info(f"Successfully installed {component} via YUM")
            return True
        else:
            self.logger.error(f"Failed to install {component} via YUM: {stderr}")
            return False
    
    def _install_via_pip(self, component: str) -> bool:
        """Install component via pip"""
        packages = {
            'nvidia-ml-py': 'nvidia-ml-py3',
            'tensorrt': 'tensorrt',
        }
        
        if component not in packages:
            self.logger.error(f"Pip installation not supported for {component}")
            return False
        
        success, stdout, stderr = self._run_command(['pip3', 'install', '--upgrade', packages[component]])
        if success:
            self.logger.info(f"Successfully installed {component} via pip")
            return True
        else:
            self.logger.error(f"Failed to install {component} via pip: {stderr}")
            return False
    
    def _install_via_conda(self, component: str) -> bool:
        """Install component via conda"""
        packages = {
            'cuda': 'cudatoolkit',
            'cudnn': 'cudnn',
            'tensorrt': 'tensorrt',
        }
        
        if component not in packages:
            self.logger.error(f"Conda installation not supported for {component}")
            return False
        
        success, stdout, stderr = self._run_command(['conda', 'install', '-y', packages[component]])
        if success:
            self.logger.info(f"Successfully installed {component} via conda")
            return True
        else:
            self.logger.error(f"Failed to install {component} via conda: {stderr}")
            return False
    
    def _install_manual(self, component: str) -> bool:
        """Provide manual installation instructions"""
        self.logger.info(f"Manual installation required for {component}")
        
        instructions = {
            'driver': "Visit https://www.nvidia.com/drivers/ and download the appropriate driver for your system",
            'cuda': "Visit https://developer.nvidia.com/cuda-downloads and download the CUDA toolkit",
            'cudnn': "Visit https://developer.nvidia.com/cudnn and download cuDNN library",
            'tensorrt': "Visit https://developer.nvidia.com/tensorrt and download TensorRT",
        }
        
        if component in instructions:
            print(f"\nManual installation instructions for {component}:")
            print(instructions[component])
            return True
        
        return False
    
    def generate_update_report(self, updates: List[UpdateInfo]) -> str:
        """Generate a report of available updates"""
        report = []
        report.append("=" * 80)
        report.append("NVIDIA SOFTWARE UPDATE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        report.append("")
        
        available_updates = [u for u in updates if u.update_available]
        no_updates = [u for u in updates if not u.update_available]
        
        if available_updates:
            report.append("AVAILABLE UPDATES")
            report.append("-" * 40)
            for update in available_updates:
                report.append(f"Component: {update.component}")
                report.append(f"  Current Version: {update.current_version}")
                report.append(f"  Latest Version: {update.latest_version}")
                report.append(f"  Compatibility: {update.compatibility}")
                report.append(f"  Download URL: {update.download_url}")
                report.append("")
        
        if no_updates:
            report.append("UP TO DATE COMPONENTS")
            report.append("-" * 40)
            for update in no_updates:
                report.append(f"{update.component}: {update.current_version} (Latest)")
            report.append("")
        
        if not available_updates and not no_updates:
            report.append("No components checked for updates.")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point for update manager"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NVIDIA Update Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                    # Check for available updates
  %(prog)s --install driver          # Install/update NVIDIA driver
  %(prog)s --install cuda --method apt # Install CUDA via APT
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check for available updates'
    )
    
    parser.add_argument(
        '--install',
        type=str,
        help='Install/update specified component'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='auto',
        choices=['auto', 'apt', 'yum', 'pip', 'conda', 'manual'],
        help='Installation method'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create update manager instance
    update_manager = NVIDIAUpdateManager(verbose=args.verbose)
    
    try:
        if args.check:
            # Check for updates
            updates = update_manager.check_updates()
            report = update_manager.generate_update_report(updates)
            print(report)
        
        elif args.install:
            # Install component
            success = update_manager.install_component(args.install, args.method)
            if success:
                print(f"Successfully installed/updated {args.install}")
            else:
                print(f"Failed to install/update {args.install}")
                sys.exit(1)
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
