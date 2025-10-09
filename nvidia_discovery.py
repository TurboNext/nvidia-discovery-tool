#!/usr/bin/env python3
"""
NVIDIA Hardware Discovery and Management Tool
A comprehensive tool for discovering NVIDIA hardware and managing software updates
across Unix variants in AI training and inference environments.

Adds one-line hardware summaries:
- GPU: "4x4090 + 2xH100"
- CPU: "2 x AMD EPYC 9555"
- Storage: "4 x 1.6TB PCIe 5 NVMe SSD (Solidigm PS1030 1.6T)"
- NIC: "Dual-port 200GbE (Nvidia MCX755106AS-HEAT)"
- RAM: "24 x 16GB DDR5 6400 Mhz"
- System disk: "2 x SATA 3.84T SSD (Solidigm S4520)"
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

# ---------------- Dataclasses ----------------

@dataclass
class GPUInfo:
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
    name: str
    version: str
    path: str
    status: str

@dataclass
class SystemInfo:
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

# ---------------- Helper formatting ----------------

def _first_int(s: str, default: int = 0) -> int:
    m = re.search(r'\d+', s or '')
    return int(m.group()) if m else default

def _first_float(s: str, default: float = 0.0) -> float:
    m = re.search(r'(\d+(\.\d+)?)', s or '')
    return float(m.group(1)) if m else default

def _normalize_size_gb(size_str: str) -> str:
    """Normalize lsblk/nvme sizes into TB with 1 decimal if >= 1 TB else in GB."""
    s = size_str.strip().upper()
    # Already human (e.g., "1.6T", "3.84T", "931.5G")
    if re.search(r'[TGMK]B?$', s):
        return s.replace('IB', 'B')
    # Bytes
    try:
        v = float(size_str)
        if v >= 1e12:
            return f"{v/1e12:.2f}T"
        elif v >= 1e9:
            return f"{v/1e9:.1f}G"
        else:
            return f"{v:.0f}B"
    except:
        return size_str

def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"

# ---------------- Main class ----------------

class NVIDIADiscovery:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.system_info = None
        self.gpus: List[GPUInfo] = []
        self.software_components: List[SoftwareInfo] = []

    def _setup_logging(self) -> logging.Logger:
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

    def _run_command(self, command: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        try:
            self.logger.debug(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
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

    # ---------- System info (unchanged except CUDA parsing logs kept) ----------

    def _get_system_info(self) -> SystemInfo:
        self.logger.info("Gathering system information...")
        hostname = platform.node()
        os_name = platform.system()
        os_version = platform.release()
        kernel_version = platform.version()
        architecture = platform.machine()

        success, uname_output, _ = self._run_command(['uname', '-a'])
        uname_output = uname_output.strip() if success else "Unknown"

        distribution = self._get_distribution()
        package_managers = self._discover_package_managers()
        disk_info = self._get_disk_info()

        success, network_info, _ = self._run_command(['ip', 'a'])
        network_info = network_info.strip() if success else "Unknown"

        success, stdout, _ = self._run_command(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits']
        )
        nvidia_driver_version = stdout.strip().split('\n')[0] if success and stdout.strip() else "Unknown"

        cuda_runtime_version = "Unknown"
        success, stdout, _ = self._run_command(['nvidia-smi', '--version'])
        if success and stdout:
            self.logger.info(f"nvidia-smi --version output: {repr(stdout)}")
            for line in stdout.split('\n'):
                line = line.strip()
                self.logger.info(f"Processing line: {repr(line)}")
                if 'CUDA Version' in line and ':' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        cuda_runtime_version = parts[1].strip().split()[0]
                        break
                elif 'CUDA' in line and any(ch.isdigit() for ch in line):
                    m = re.search(r'CUDA\s+(\d+\.\d+)', line)
                    if m:
                        cuda_runtime_version = m.group(1)
                        break
                elif 'CUDA' in line:
                    m = re.search(r'(\d+\.\d+)', line)
                    if m:
                        cuda_runtime_version = m.group(1)
                        break

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

    # ---------- GPU discovery (unchanged) ----------

    def _parse_gpu_info(self, nvidia_smi_output: str) -> List[GPUInfo]:
        gpus = []
        lines = nvidia_smi_output.strip().split('\n')
        if not lines or not lines[0].strip():
            return gpus
        for line in lines:
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 13:
                try:
                    gpus.append(GPUInfo(
                        index=int(parts[0]) if parts[0].isdigit() else 0,
                        name=parts[1] if len(parts) > 1 else "Unknown",
                        driver_version=parts[2] if len(parts) > 2 else "Unknown",
                        cuda_version="Unknown",
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
                        compute_capability="Unknown"
                    ))
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing GPU line: {line} - {e}")
        return gpus

    def discover_gpus(self) -> List[GPUInfo]:
        self.logger.info("Discovering NVIDIA GPUs...")
        success, _, _ = self._run_command(['nvidia-smi', '--version'])
        if not success:
            self.logger.error("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
            return []
        success, stdout, stderr = self._run_command([
            'nvidia-smi',
            '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,uuid,pci.bus_id',
            '--format=csv,noheader,nounits'
        ])
        if not success:
            self.logger.error(f"Failed to query GPU information: {stderr}")
            return []
        gpus = self._parse_gpu_info(stdout)
        for gpu in gpus:
            self._enrich_gpu_info(gpu)
        self.logger.info(f"Discovered {len(gpus)} GPU(s)")
        return gpus

    def _enrich_gpu_info(self, gpu: GPUInfo) -> None:
        try:
            success, stdout, _ = self._run_command([
                'nvidia-smi', f'--id={gpu.index}',
                '--query-gpu=compute_cap', '--format=csv,noheader,nounits'
            ])
            if success and stdout.strip():
                gpu.compute_capability = stdout.strip()
        except Exception as e:
            self.logger.debug(f"Could not enrich GPU {gpu.index} info: {e}")

    # ---------- Software (unchanged content) ----------

    def discover_software_components(self) -> List[SoftwareInfo]:
        self.logger.info("Discovering NVIDIA software components...")
        components = []
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
        for tool, _ in nvidia_tools:
            version = self._get_tool_version(tool)
            path = self._find_tool_path(tool)
            status = "Available" if version != "Not found" else "Not found"
            components.append(SoftwareInfo(name=tool, version=version, path=path, status=status))
        components.extend(self._discover_cuda_libraries())
        components.extend(self._discover_python_packages())
        components.extend(self._discover_container_support())
        components.extend(self._discover_appimage_support())
        return components

    def _get_tool_version(self, tool: str) -> str:
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
                m = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                return m.group(1) if m else stdout.strip().split('\n')[0]
        return "Not found"

    def _find_tool_path(self, tool: str) -> str:
        success, stdout, _ = self._run_command(['which', tool])
        return stdout.strip() if success and stdout.strip() else "Not found"

    def _discover_cuda_libraries(self) -> List[SoftwareInfo]:
        components = []
        cuda_paths = ['/usr/local/cuda', '/opt/cuda', '/usr/cuda', '/usr/local/cuda-*']
        for base_path in cuda_paths:
            paths = []
            if '*' in base_path:
                import glob
                paths = glob.glob(base_path)
            elif os.path.exists(base_path):
                paths = [base_path]
            for path in paths:
                if os.path.isdir(path):
                    lib_path = os.path.join(path, 'lib64')
                    if os.path.exists(lib_path):
                        for lib_name in ['libcudnn.so', 'libcublas.so', 'libcurand.so']:
                            lib_file = os.path.join(lib_path, lib_name)
                            if os.path.exists(lib_file):
                                components.append(SoftwareInfo(
                                    name=f"{lib_name} ({path})",
                                    version=self._get_library_version(lib_file),
                                    path=lib_file,
                                    status="Available"
                                ))
        return components

    def _get_library_version(self, lib_path: str) -> str:
        try:
            success, stdout, _ = self._run_command(['ldd', lib_path])
            if success:
                m = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                if m: return m.group(1)
        except:
            pass
        return "Unknown"

    def _discover_python_packages(self) -> List[SoftwareInfo]:
        return []

    def _discover_container_support(self) -> List[SoftwareInfo]:
        components = []
        success, stdout, _ = self._run_command(['docker', '--version'])
        if success:
            m = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = m.group(1) if m else "Unknown"
            components.append(SoftwareInfo(
                name="docker", version=version, path=self._find_tool_path('docker'), status="Available"
            ))
        success, _, _ = self._run_command(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi'])
        components.append(SoftwareInfo(
            name="nvidia-docker-support", version="N/A", path="Docker runtime",
            status="Available" if success else "Not available"
        ))
        success, stdout, _ = self._run_command(['podman', '--version'])
        if success:
            m = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
            version = m.group(1) if m else "Unknown"
            components.append(SoftwareInfo(
                name="podman", version=version, path=self._find_tool_path('podman'), status="Available"
            ))
        components.extend(self._discover_additional_tools())
        components.extend(self._discover_nvidia_ctk_config())
        return components

    def _discover_additional_tools(self) -> List[SoftwareInfo]:
        components = []
        for tool in ['nvidia-ctk','nvidia-container-cli','nvidia-cuda-mps-server','nvidia-cuda-mps-control']:
            success, stdout, _ = self._run_command([tool, '--version'])
            if success:
                m = re.search(r'(\d+\.\d+(?:\.\d+)?)', stdout)
                version = m.group(1) if m else "Unknown"
                components.append(SoftwareInfo(
                    name=tool, version=version, path=self._find_tool_path(tool), status="Available"
                ))
        return components

    def _discover_nvidia_ctk_config(self) -> List[SoftwareInfo]:
        components = []
        success, stdout, _ = self._run_command(['which', 'nvidia-ctk'])
        if success and stdout.strip():
            nvidia_ctk_path = stdout.strip()
            success, ver_out, _ = self._run_command(['nvidia-ctk', '--version'])
            version = "Unknown"
            if success and ver_out:
                m = re.search(r'(\d+\.\d+(?:\.\d+)?)', ver_out)
                if m: version = m.group(1)
            components.append(SoftwareInfo(name="nvidia-ctk", version=version, path=nvidia_ctk_path, status="Available"))
            for cfg in ['/etc/nvidia-container-runtime/config.toml',
                        '/usr/local/nvidia-toolkit/config.toml',
                        '/etc/nvidia-container-toolkit/config.toml']:
                if os.path.exists(cfg):
                    try:
                        with open(cfg, 'r') as f:
                            f.read()
                        components.append(SoftwareInfo(
                            name=f"nvidia-ctk-config ({cfg})", version="Config", path=cfg, status="Available"
                        ))
                    except Exception as e:
                        if self.verbose: self.logger.warning(f"Could not read {cfg}: {e}")
        success, stdout, _ = self._run_command(['which', 'nvidia-container-cli'])
        if success and stdout.strip():
            components.append(SoftwareInfo(
                name="nvidia-container-cli", version="Available", path=stdout.strip(), status="Available"
            ))
        return components

    def _discover_appimage_support(self) -> List[SoftwareInfo]:
        components = []
        success, stdout, _ = self._run_command(['which', 'appimagelauncher'])
        if success and stdout.strip():
            success, ver_out, _ = self._run_command(['appimagelauncher', '--version'])
            version = "Unknown"
            if success and ver_out:
                m = re.search(r'(\d+\.\d+(?:\.\d+)?)', ver_out)
                if m: version = m.group(1)
            components.append(SoftwareInfo(name="appimagelauncher", version=version, path=stdout.strip(), status="Available"))
        success, stdout, _ = self._run_command(['which', 'appimaged'])
        if success and stdout.strip():
            components.append(SoftwareInfo(name="appimaged", version="Runtime", path=stdout.strip(), status="Available"))
        appimage_dirs = [os.path.expanduser('~/Applications'), os.path.expanduser('~/bin'), '/opt/appimages', '/usr/local/bin']
        found = []
        for d in appimage_dirs:
            if os.path.exists(d):
                try:
                    for f in os.listdir(d):
                        if f.endswith(('.AppImage', '.appimage')):
                            full = os.path.join(d, f)
                            if os.path.isfile(full) and os.access(full, os.X_OK):
                                found.append(full)
                except (PermissionError, OSError):
                    pass
        for p in found[:10]:
            components.append(SoftwareInfo(name=f"appimage-{os.path.basename(p)}", version="AppImage", path=p, status="Available"))
        success, stdout, _ = self._run_command(['which', 'xdg-desktop-portal'])
        if success and stdout.strip():
            components.append(SoftwareInfo(name="xdg-desktop-portal", version="Available", path=stdout.strip(), status="Available"))
        return components

    # ---------- OS / PM / Disks (mostly unchanged) ----------

    def _get_distribution(self) -> str:
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=')[1].strip().strip('"')
                    if line.startswith('NAME='):
                        return line.split('=')[1].strip().strip('"')
        except:
            pass
        return "Unknown"

    def _discover_package_managers(self) -> List[Dict[str, str]]:
        package_managers = []
        pm_commands = [
            ('apt', 'apt --version'), ('apt-get', 'apt-get --version'), ('dpkg', 'dpkg --version'), ('aptitude', 'aptitude --version'),
            ('yum', 'yum --version'), ('dnf', 'dnf --version'), ('rpm', 'rpm --version'),
            ('zypper', 'zypper --version'), ('pacman', 'pacman --version'), ('portage', 'emerge --version'),
            ('snap', 'snap --version'), ('flatpak', 'flatpak --version'), ('appimage', 'appimage --version'), ('nix', 'nix --version'),
            ('pip', 'pip --version'), ('pip3', 'pip3 --version'), ('conda', 'conda --version'),
            ('npm', 'npm --version'), ('yarn', 'yarn --version'), ('gem', 'gem --version'),
            ('cargo', 'cargo --version'), ('go', 'go version'), ('docker', 'docker --version'), ('podman', 'podman --version'),
        ]
        for pm_name, cmd in pm_commands:
            success, stdout, _ = self._run_command(cmd.split())
            if success:
                version = self._extract_version_from_output(stdout, pm_name)
                package_managers.append({'name': pm_name, 'version': version})
        return package_managers

    def _extract_version_from_output(self, output: str, pm_name: str) -> str:
        if not output: return "Unknown"
        m = re.search(r'(\d+\.\d+(?:\.\d+)?)', output)
        return m.group(1) if m else output.split('\n')[0].strip()

    def _get_disk_info(self) -> Dict[str, Any]:
        disk_info: Dict[str, Any] = {}
        success, stdout, _ = self._run_command(['df', '-h'])
        if success: disk_info['df_output'] = stdout.strip()
        success, stdout, _ = self._run_command(['lsblk'])
        if success: disk_info['lsblk_output'] = stdout.strip()
        success, stdout, _ = self._run_command(['lsblk', '-o', 'NAME,MODEL,SIZE,TYPE,ROTA,TRAN'])
        if success: disk_info['lsblk_detailed'] = stdout.strip()
        success, stdout, _ = self._run_command(['sh', '-c', 'cat /sys/block/sd*/queue/rotational 2>/dev/null || echo "No rotational info available"'])
        if success: disk_info['rotational_info'] = stdout.strip()
        success, stdout, _ = self._run_command(['blkid'])
        if success: disk_info['disk_formats'] = stdout.strip()
        success, stdout, _ = self._run_command(['df', '-h', '/'])
        if success:
            lines = stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    disk_info['root_filesystem'] = {
                        'device': parts[0], 'size': parts[1], 'used': parts[2],
                        'available': parts[3],
                        'use_percent': parts[4] if len(parts) > 4 else 'Unknown',
                        'mounted_on': parts[5] if len(parts) > 5 else 'Unknown'
                    }
        return disk_info

    # ---------- New: One-line hardware summaries ----------

    def _short_gpu_model(self, full: str) -> str:
        """
        Convert 'NVIDIA GeForce RTX 4090' -> '4090', 'NVIDIA H100 PCIe' -> 'H100', 'Tesla T4' -> 'T4'
        """
        if not full: return "GPU"
        # Prefer H100/H200/B200/L40/L4/T4/A100/A800/4090/4090D/4080 etc.
        m = re.search(r'(H100|H200|B100|B200|L40S?|L4|T4|A100|A800|A30|A10|RTX?\s*4090D?|RTX?\s*4090|RTX?\s*4080|RTX?\s*4070Ti?|RTX?\s*4070|4090D|4090|4080|4070Ti?|\bV100\b|\bP100\b|\bK80\b)', full, re.IGNORECASE)
        if m:
            token = m.group(1).upper().replace("RTX", "").strip()
            token = token.replace("  ", " ")
            return token
        # Fallback: last token with digits/letters
        toks = [t for t in re.split(r'[\s\-]+', full) if re.search(r'[A-Za-z0-9]', t)]
        for t in reversed(toks):
            if re.search(r'\d', t):
                return t.upper()
        return full.strip().split()[0]

    def _gpu_summary_line(self, gpus: List[Dict[str, Any]]) -> str:
        counter = Counter(self._short_gpu_model(g['name']) for g in gpus if g.get('name'))
        parts = [f"{count}x{model}" for model, count in sorted(counter.items(), key=lambda x: x[0])]
        return " + ".join(parts) if parts else "None detected"

    def _cpu_summary_line(self) -> str:
        """
        Try lscpu first. If available, use 'Socket(s)' and 'Model name'.
        Fallback to /proc/cpuinfo uniqueness on 'physical id' and 'model name'.
        """
        # lscpu (pretty robust)
        ok, out, _ = self._run_command(['lscpu'])
        sockets = None
        model = None
        if ok and out:
            for line in out.splitlines():
                if 'Socket(s):' in line:
                    sockets = _first_int(line)
                elif 'Model name:' in line:
                    model = line.split(':', 1)[1].strip()
        if sockets and model:
            # Compress to vendor + family like "AMD EPYC 9555" if present
            m = re.search(r'(AMD|Intel).+', model, re.IGNORECASE)
            nice = m.group(0).strip() if m else model
            # Strip CPU @ freq / (R) (TM)
            nice = re.sub(r'@.*', '', nice)
            nice = nice.replace('(R)', '').replace('(TM)', '').replace('CPU', '').strip()
            return f"{sockets} x {nice}"

        # /proc/cpuinfo fallback
        try:
            with open('/proc/cpuinfo','r') as f:
                txt = f.read()
            phys_ids = set()
            model_name = None
            for blk in txt.strip().split('\n\n'):
                d = {}
                for ln in blk.splitlines():
                    if ':' in ln:
                        k,v = [p.strip() for p in ln.split(':',1)]
                        d[k]=v
                if 'physical id' in d: phys_ids.add(d['physical id'])
                if not model_name and 'model name' in d: model_name = d['model name']
            sockets = len(phys_ids) if phys_ids else 1
            if model_name:
                nice = re.sub(r'@.*','',model_name)
                nice = nice.replace('(R)','').replace('(TM)','').replace('CPU','').strip()
                return f"{sockets} x {nice}"
        except:
            pass
        return "Unknown"

    def _ram_summary_line(self) -> str:
        """
        dmidecode gives per-DIMM size and speed -> '24 x 16GB DDR5 6400 Mhz'
        Falls back to MemTotal only if dmidecode unavailable.
        """
        # dmidecode requires sudo
        ok, out, _ = self._run_command(['dmidecode', '--type', 'memory'])
        if ok and out:
            dimms = []
            mem_type = None
            speed_mhz = None
            for sec in out.split('\n\n'):
                if 'Memory Device' in sec and 'Size:' in sec:
                    size = None
                    for ln in sec.splitlines():
                        if ln.strip().startswith('Size:'):
                            s = ln.split(':',1)[1].strip()
                            if s.lower() != 'no module installed' and s.lower() != 'not installed':
                                size = s
                        if ln.strip().startswith('Type:') and ('DDR' in ln or 'LP' in ln):
                            mem_type = ln.split(':',1)[1].strip()
                        if ln.strip().startswith('Configured Memory Speed:'):
                            v = _first_int(ln)
                            if v: speed_mhz = v
                    if size:
                        dimms.append(size)
            if dimms:
                # size like "16384 MB" or "16 GB"
                sizes_gb = []
                for s in dimms:
                    m = re.search(r'(\d+)\s*MB', s, re.IGNORECASE)
                    if m: sizes_gb.append(int(m.group(1))/1024)
                    else:
                        m = re.search(r'(\d+)\s*GB', s, re.IGNORECASE)
                        if m: sizes_gb.append(int(m.group(1)))
                # If uniform DIMMs (common): count x first_size
                if sizes_gb and all(abs(x - sizes_gb[0]) < 0.1 for x in sizes_gb):
                    count = len(sizes_gb)
                    size_each = int(round(sizes_gb[0]))
                    typ = mem_type or "DDR5"
                    spd = f" {speed_mhz} Mhz" if speed_mhz else ""
                    return f"{count} x {size_each}GB {typ}{spd}"
        # Fallback: total only
        try:
            with open('/proc/meminfo','r') as f:
                for ln in f:
                    if ln.startswith('MemTotal:'):
                        kb = _first_int(ln)
                        gb = kb/1024/1024
                        return f"Total {gb:.1f}GB (slot details require sudo dmidecode)"
        except:
            pass
        return "Unknown"

    def _nvme_storage_summary_line(self) -> str:
        """
        Summarize non-system NVMe SSDs by (model, size, pcie gen if detectable).
        Uses `nvme list -o json` if available, else lsblk -d.
        Format: '4 x 1.6TB PCIe 5 NVMe SSD (Solidigm PS1030 1.6T)'
        """
        # Try nvme json for gen/speed
        ok, out, _ = self._run_command(['nvme', 'list', '-o', 'json'])
        entries = []
        if ok and out:
            try:
                data = json.loads(out)
                for ns in data.get('Devices', []):
                    if ns.get('Name','').startswith('/dev/nvme'):
                        model = (ns.get('ModelNumber') or ns.get('ModelNumberHumanReadable') or '').strip()
                        size = _normalize_size_gb(ns.get('PhysicalSize',''))
                        # Some nvme tools include "PciGen" or "PciLanes"
                        pcie_gen = ns.get('PciGen') or ns.get('PciGeneration') or ''
                        if isinstance(pcie_gen, dict):
                            pcie_gen = pcie_gen.get('Current') or pcie_gen.get('Max') or ''
                        pcie_txt = f"PCIe {pcie_gen}" if str(pcie_gen).strip() else "PCIe"
                        entries.append((model, size, pcie_txt))
            except Exception as e:
                self.logger.debug(f"nvme json parse failed: {e}")

        if not entries:
            # Fallback to lsblk -d for model/size; TRAN 'nvme'
            ok, out, _ = self._run_command(['lsblk', '-d', '-o', 'NAME,MODEL,SIZE,TYPE,TRAN'])
            if ok and out:
                for ln in out.splitlines()[1:]:
                    parts = ln.split()
                    if len(parts) >= 5:
                        name, model, size, typ, tran = parts[0], " ".join(parts[1:-3+2]), parts[-3], parts[-2], parts[-1]
                        if typ.lower() == 'disk' and tran.lower() == 'nvme':
                            entries.append((model.strip(), _normalize_size_gb(size), "PCIe"))

        if not entries:
            return "None detected"

        # Group by (model normalized, size, pcie)
        grouped: Dict[Tuple[str,str,str], int] = Counter((m or "NVMe SSD", s or "", p) for (m,s,p) in entries)
        parts = []
        for (model, size, pcie_txt), cnt in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            size_disp = size if size else ""
            suffix = f"{pcie_txt} NVMe SSD ({model})" if model else f"{pcie_txt} NVMe SSD"
            if size_disp:
                parts.append(f"{cnt} x {size_disp} {suffix}")
            else:
                parts.append(f"{cnt} x {suffix}")
        return " + ".join(parts)

    def _system_disk_summary_line(self) -> str:
        """
        Try to identify system/root device and its mirror/peers (common /dev/sd* SATA SSDs).
        Format example: '2 x SATA 3.84T SSD (Solidigm S4520)'
        """
        # Find root device from df /
        root_dev = None
        ok, out, _ = self._run_command(['df', '-h', '/'])
        if ok and out:
            lines = out.strip().splitlines()
            if len(lines) >= 2:
                root_dev = lines[1].split()[0]

        # Map block devices via lsblk -o NAME,MODEL,SIZE,TYPE,ROTA,TRAN
        ok, out, _ = self._run_command(['lsblk', '-o', 'NAME,MODEL,SIZE,TYPE,ROTA,TRAN'])
        devices = []
        if ok and out:
            header = True
            for ln in out.splitlines():
                if header: header=False; continue
                cols = ln.split()
                if len(cols) >= 6:
                    name, model, size, typ, rota, tran = cols[0], " ".join(cols[1:-4+1]), cols[-4], cols[-3], cols[-2], cols[-1]
                    if typ.lower() == 'disk':
                        devices.append((name, model.strip(), _normalize_size_gb(size), tran.upper()))
        # Heuristic: system disks are SATA SSDs (TRAN == 'SATA' or 'ATA') that hold root or are smallest SSD pair
        sata_like = [(n,m,s,t) for (n,m,s,t) in devices if t in ('SATA','ATA')]
        if not sata_like:
            # maybe NVMe used for system
            nvmes = [(n,m,s,t) for (n,m,s,t) in devices if t == 'NVME']
            if nvmes:
                # choose smallest 1-2 for system
                try:
                    sorted_nvmes = sorted(nvmes, key=lambda x: _first_float(x[2]))
                except:
                    sorted_nvmes = nvmes
                pick = sorted_nvmes[:2]
                count = len(pick)
                if count >= 1:
                    model = pick[0][1] or "NVMe SSD"
                    size = pick[0][2]
                    return f"{count} x {size} NVMe SSD ({model})"
                return "Unknown"
            return "Unknown"

        # If we can detect a model and size from SATA list, assume mirrored pair if >=2 identical
        group = Counter((m or "SATA SSD", s) for (_n,m,s,_t) in sata_like)
        # choose most common group
        if group:
            (model, size), cnt = group.most_common(1)[0]
            # Try to include explicit 'SATA' in text
            return f"{cnt} x SATA {size} SSD ({model})"
        return "Unknown"

    def _nic_summary_line(self) -> str:
        """
        Attempt: aggregate NICs by PCI device (lspci), infer port count and max speed via ethtool.
        Example: 'Dual-port 200GbE (Nvidia MCX755106AS-HEAT)'
        """
        # Map PCI ethernet controllers
        ok, out, _ = self._run_command(['lspci'])
        pci_eth = []
        if ok and out:
            for ln in out.splitlines():
                if re.search(r'ethernet controller', ln, re.IGNORECASE):
                    pci_eth.append(ln.strip())

        # If none, fallback
        if not pci_eth:
            return "Unknown"

        # Pick Mellanox/NVIDIA line if present
        preferred = None
        for ln in pci_eth:
            if re.search(r'(Mellanox|NVIDIA|NVAIE|MCX)', ln, re.IGNORECASE):
                preferred = ln
                break
        model_txt = preferred or pci_eth[0]
        # Extract model-ish token in parentheses if available
        paren = re.search(r'\(([^)]+)\)', model_txt)
        model = paren.group(1) if paren else model_txt.split(':')[-1].strip()

        # Interfaces and speeds (needs ethtool)
        ok, links, _ = self._run_command(['ip', '-o', 'link', 'show'])
        if not ok or not links:
            return f"{model}"

        ifnames = []
        for ln in links.splitlines():
            m = re.match(r'\d+:\s*([^:]+):', ln)
            if m:
                name = m.group(1)
                if not name.startswith(('lo','docker','veth','cni','br-','virbr','tap','wg','tun')):
                    ifnames.append(name)

        # Query speeds
        speeds = []
        for iface in ifnames:
            ok, out, _ = self._run_command(['ethtool', iface])
            if ok and out:
                spm = re.search(r'Speed:\s*([0-9]+)Mb/s', out)
                if spm:
                    speeds.append(int(spm.group(1)))
        # Determine port count and headline speed (unique physical may be hard; use number of up to N ports with same top speed)
        if speeds:
            top = max(speeds)
            ports = sum(1 for s in speeds if s == top)
            # Map number to wording
            port_word = {1:"Single-port", 2:"Dual-port", 4:"Quad-port"}.get(ports, f"{ports}-port")
            # 200000 Mb/s -> 200GbE
            gbit = top // 1000
            return f"{port_word} {gbit}GbE ({model})"

        # If no ethtool speed, just show model
        return f"{model}"

    # ---------- Orchestration ----------

    def run_discovery(self) -> Dict[str, Any]:
        self.logger.info("Starting NVIDIA discovery process...")
        self.system_info = self._get_system_info()
        self.gpus = self.discover_gpus()
        self.software_components = self.discover_software_components()
        return {
            'system_info': asdict(self.system_info),
            'gpus': [asdict(gpu) for gpu in self.gpus],
            'software_components': [asdict(comp) for comp in self.software_components],
            'discovery_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }

    # ---------- Report generation (extended) ----------

    def generate_report(self, data: Dict[str, Any], format: str = 'text') -> str:
        if format == 'json':
            return json.dumps(data, indent=2)

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

        report.append("SYSTEM DETAILS")
        report.append("-" * 40)
        report.append(f"Uname: {sys_info['uname_output']}")
        report.append("")

        # ---------------- NEW: Hardware Summaries ----------------
        report.append("HARDWARE SUMMARIES")
        report.append("-" * 40)
        gpus = data.get('gpus', [])
        report.append(f"GPU summary: {self._gpu_summary_line(gpus)}")
        report.append(f"CPU summary: {self._cpu_summary_line()}")
        report.append(f"Storage summary: {self._nvme_storage_summary_line()}")
        report.append(f"NIC summary: {self._nic_summary_line()}")
        report.append(f"RAM summary: {self._ram_summary_line()}")
        report.append(f"System disk summary: {self._system_disk_summary_line()}")
        report.append("")

        # Package Managers
        if sys_info['package_managers']:
            report.append("PACKAGE MANAGERS")
            report.append("-" * 40)
            for pm in sys_info['package_managers']:
                report.append(f"  - {pm['name']}: {pm['version']}")
            report.append("")

        # Disk Information (existing detail sections retained)
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
            if 'lsblk_detailed' in sys_info['disk_info']:
                report.append("Disk Devices (lsblk -o NAME,MODEL,SIZE,TYPE,ROTA,TRAN):")
                for line in sys_info['disk_info']['lsblk_detailed'].split('\n'):
                    report.append(f"  {line}")
                report.append("")
            if 'rotational_info' in sys_info['disk_info']:
                report.append("Rotational Information (cat /sys/block/sd*/queue/rotational):")
                for line in sys_info['disk_info']['rotational_info'].split('\n'):
                    report.append(f"  {line}")
                report.append("")
            if 'disk_formats' in sys_info['disk_info']:
                report.append("Disk Formats (blkid):")
                for line in sys_info['disk_info']['disk_formats'].split('\n'):
                    report.append(f"  {line}")
                report.append("")
            if 'df_output' in sys_info['disk_info']:
                report.append("Disk Usage (df -h):")
                for line in sys_info['disk_info']['df_output'].split('\n'):
                    report.append(f"  {line}")
                report.append("")

        # Network Information (verbatim)
        if 'network_info' in sys_info and sys_info['network_info'] != "Unknown":
            report.append("NETWORK INFORMATION")
            report.append("-" * 40)
            for line in sys_info['network_info'].split('\n'):
                if line.strip():
                    report.append(f"  {line}")
            report.append("")

        # GPU Summary (existing block retained)
        if gpus:
            report.append("GPU SUMMARY")
            report.append("-" * 40)
            gpu_combinations = {}
            for gpu in gpus:
                key = f"{gpu['name']} (Driver: {gpu['driver_version']})"
                gpu_combinations[key] = gpu_combinations.get(key, 0) + 1
            for combination, count in sorted(gpu_combinations.items()):
                report.append(f"  {combination}: {count} GPU(s)")
            report.append("")

        # GPU Details
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

        # Software Components
        components = data['software_components']
        report.append("SOFTWARE COMPONENTS")
        report.append("-" * 40)
        report.append("Component,Version,Status,Path")
        for comp in sorted(components, key=lambda x: x['name']):
            path = comp['path'] if comp['path'] != "Not found" else "N/A"
            report.append(f"{comp['name']},{comp['version']},{comp['status']},{path}")
        report.append("")

        # Thresholds (kept as-is)
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

    # ---------- Thresholds (unchanged) ----------

    def _load_thresholds(self) -> List[Dict[str, str]]:
        thresholds = []
        csv_path = os.path.join(os.path.dirname(__file__), 'thresholds.csv')
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
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
        results = []
        if self.verbose: self.logger.info(f"Checking {len(thresholds)} thresholds")
        for threshold in thresholds:
            component = threshold['component']
            threshold_value = threshold['threshold']
            operator = threshold['operator']
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
        return results

    def _get_measured_value(self, data: Dict[str, Any], component: str) -> str:
        if component == 'cuda_version':
            gpus = data.get('gpus', [])
            if gpus and len(gpus) > 0:
                cuda_version = gpus[0].get('cuda_version', 'Unknown')
                if cuda_version != 'Unknown':
                    return cuda_version
            system_cuda = data.get('system_info', {}).get('cuda_runtime_version', 'Unknown')
            if system_cuda != 'Unknown':
                return system_cuda
            success, stdout, _ = self._run_command(['nvidia-smi', '--version'])
            if success and stdout:
                for line in stdout.split('\n'):
                    line = line.strip()
                    if 'CUDA Version' in line and ':' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            return parts[1].strip().split()[0]
            return 'Unknown'
        elif component == 'cudnn':
            for comp in data.get('software_components', []):
                if comp.get('name') == 'cudnn':
                    return comp.get('version', 'Unknown')
            return 'Unknown'
        return "Unknown"

    def _compare_versions(self, measured: str, threshold: str, operator: str) -> bool:
        if measured in ("Unknown", "Not found"):
            return False
        try:
            mp = [int(x) for x in measured.split('.')]
            tp = [int(x) for x in threshold.split('.')]
            L = max(len(mp), len(tp))
            mp.extend([0]*(L-len(mp)))
            tp.extend([0]*(L-len(tp)))
            if operator == '>=': return mp >= tp
            if operator == '>':  return mp > tp
            if operator == '<=': return mp <= tp
            if operator == '<':  return mp < tp
            if operator == '==': return mp == tp
            return False
        except:
            return False

# ---------------- CLI ----------------

def main():
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
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--verbose','-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output','-o', type=str, help='Output file path (default: stdout)')
    parser.add_argument('--check-updates', action='store_true', help='Check for available updates')
    parser.add_argument('--update', type=str, choices=['driver','cuda','cudnn','tensorrt'], help='Update specified component')
    parser.add_argument('--update-method', type=str, default='auto',
                        choices=['auto','apt','yum','pip','conda','manual'], help='Update method to use')
    args = parser.parse_args()

    discovery = NVIDIADiscovery(verbose=args.verbose)

    try:
        if args.check_updates or args.update:
            from update_manager import NVIDIAUpdateManager
            update_manager = NVIDIAUpdateManager(verbose=args.verbose)
            if args.check_updates:
                updates = update_manager.check_updates()
                print(update_manager.generate_update_report(updates))
            elif args.update:
                success = update_manager.install_component(args.update, args.update_method)
                print(f"Successfully updated {args.update}" if success else f"Failed to update {args.update}")
                if not success: sys.exit(1)
        else:
            data = discovery.run_discovery()
            report = discovery.generate_report(data, 'json' if args.json else 'text')
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
            import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
