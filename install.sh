#!/bin/bash
"""
Installation script for NVIDIA Discovery and Management Tool
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            VER=$VERSION_ID
        else
            OS=$(uname -s)
            VER=$(uname -r)
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
        VER=$(sw_vers -productVersion)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv curl wget
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
        if command_exists dnf; then
            sudo dnf install -y python3 python3-pip curl wget
        else
            sudo yum install -y python3 python3-pip curl wget
        fi
    elif [[ "$OS" == "macOS" ]]; then
        if ! command_exists brew; then
            print_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        brew install python3 curl wget
    else
        print_warning "Unknown OS. Please install Python 3, curl, and wget manually."
    fi
}

# Function to create virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    pip install requests urllib3
}

# Function to install NVIDIA tools
install_nvidia_tools() {
    print_status "Checking for NVIDIA tools..."
    
    if ! command_exists nvidia-smi; then
        print_warning "nvidia-smi not found. NVIDIA drivers may not be installed."
        print_status "Please install NVIDIA drivers from: https://www.nvidia.com/drivers/"
    else
        print_success "nvidia-smi found"
    fi
    
    if ! command_exists nvcc; then
        print_warning "nvcc not found. CUDA toolkit may not be installed."
        print_status "Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    else
        print_success "nvcc found"
    fi
}

# Function to create system-wide symlinks
create_symlinks() {
    print_status "Creating system-wide symlinks..."
    
    INSTALL_DIR="/usr/local/bin"
    
    if [ -w "$INSTALL_DIR" ] || sudo -n true 2>/dev/null; then
        sudo ln -sf "$(pwd)/nvidia_discovery.py" "$INSTALL_DIR/nvidia-discovery"
        sudo ln -sf "$(pwd)/update_manager.py" "$INSTALL_DIR/nvidia-update"
        print_success "Symlinks created in $INSTALL_DIR"
    else
        print_warning "Cannot create system-wide symlinks. You can run the tools directly from this directory."
    fi
}

# Function to create desktop entry (Linux only)
create_desktop_entry() {
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]] || [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
        print_status "Creating desktop entry..."
        
        DESKTOP_FILE="$HOME/.local/share/applications/nvidia-discovery.desktop"
        mkdir -p "$(dirname "$DESKTOP_FILE")"
        
        cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=NVIDIA Discovery Tool
Comment=NVIDIA Hardware Discovery and Management Tool
Exec=$(pwd)/nvidia_discovery.py
Icon=preferences-system
Terminal=true
Categories=System;Settings;
EOF
        
        chmod +x "$DESKTOP_FILE"
        print_success "Desktop entry created"
    fi
}

# Function to run tests
run_tests() {
    print_status "Running basic tests..."
    
    # Test discovery
    if python3 nvidia_discovery.py --help >/dev/null 2>&1; then
        print_success "Discovery tool test passed"
    else
        print_error "Discovery tool test failed"
        exit 1
    fi
    
    # Test update manager
    if python3 update_manager.py --help >/dev/null 2>&1; then
        print_success "Update manager test passed"
    else
        print_error "Update manager test failed"
        exit 1
    fi
}

# Main installation function
main() {
    echo "=========================================="
    echo "NVIDIA Discovery and Management Tool"
    echo "Installation Script"
    echo "=========================================="
    echo
    
    # Detect OS
    detect_os
    print_status "Detected OS: $OS $VER"
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
    
    # Install dependencies
    install_dependencies
    
    # Setup virtual environment
    setup_venv
    
    # Install NVIDIA tools
    install_nvidia_tools
    
    # Create symlinks
    create_symlinks
    
    # Create desktop entry
    create_desktop_entry
    
    # Run tests
    run_tests
    
    echo
    echo "=========================================="
    print_success "Installation completed successfully!"
    echo "=========================================="
    echo
    echo "Usage examples:"
    echo "  ./nvidia_discovery.py                    # Run discovery"
    echo "  ./nvidia_discovery.py --check-updates    # Check for updates"
    echo "  ./nvidia_discovery.py --update driver    # Update driver"
    echo "  ./nvidia_discovery.py --json             # JSON output"
    echo "  ./nvidia_discovery.py --verbose          # Verbose output"
    echo
    echo "For system-wide access, you may need to add to PATH:"
    echo "  export PATH=\$PATH:$(pwd)"
    echo
    echo "Or use the symlinks (if created):"
    echo "  nvidia-discovery --help"
    echo "  nvidia-update --help"
    echo
}

# Run main function
main "$@"
