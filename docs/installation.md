# Installation Guide

This guide provides comprehensive installation instructions for all projects in the Motion workspace.

## System Requirements

### Hardware Requirements
- **CPU:** Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **Memory:** 16GB RAM minimum, 32GB recommended for development
- **GPU:** NVIDIA GPU with 8GB+ VRAM (for deep learning training)
- **Storage:** 100GB+ free space for datasets and models

### Software Requirements
- **Operating System:** Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+
- **Python:** 3.8+ (3.9 recommended)
- **Node.js:** 16+ for web components
- **Git:** For version control and repository management

## Environment Setup

### 1. System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nodejs npm git
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.9 node git cmake
```

#### Windows
1. Install Python 3.9+ from python.org
2. Install Node.js from nodejs.org
3. Install Git from git-scm.com
4. Install Visual Studio Build Tools for C++ compilation

### 2. CUDA Setup (for GPU acceleration)

#### Check GPU Compatibility
```bash
nvidia-smi  # Should show GPU information
```

#### Install CUDA Toolkit
```bash
# Ubuntu 20.04
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Project Installation

### Clone Repository
```bash
git clone https://github.com/navichat/motion.git
cd motion
```

### 1. BvhToDeepMimic Installation

#### Option A: Install from PyPI
```bash
pip install bvhtodeepmimic
```

#### Option B: Install from Source
```bash
cd BvhToDeepMimic
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

#### Verify Installation
```bash
python -c "from bvhtomimic import BvhConverter; print('BvhToDeepMimic installed successfully')"
```

### 2. PyTorch DeepMimic Installation

#### Install Dependencies
```bash
cd pytorch_DeepMimic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==1.12.1+cu117 torchvision==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install gym==0.23.1
pip install pybullet==3.2.5
pip install mpi4py==3.1.4
pip install numpy==1.25.2
pip install matplotlib scipy

# Install package
pip install -e .
```

#### Verify Installation
```bash
cd deepmimic
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pybullet; print('PyBullet installed successfully')"
```

### 3. RSMT Installation

#### Install Dependencies
```bash
cd RSMT-Realtime-Stylized-Motion-Transition

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Alternative: Install with specific CUDA version
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch3d==0.4.0
pip install pytorch_lightning==1.5.10
pip install matplotlib==3.5.2 numpy==1.22.3 pandas==1.4.3 scipy==1.9.0
```

#### Download Dataset
```bash
# Create data directory
mkdir -p MotionData

# Download 100STYLE dataset
wget https://www.ianxmason.com/100style/100STYLE.zip
unzip 100STYLE.zip -d MotionData/
```

#### Verify Installation
```bash
python -c "import torch; import pytorch3d; print('RSMT dependencies installed successfully')"
```

### 4. Chat Interface Installation

#### Install Node.js Dependencies
```bash
# Install Psyche dependencies
cd chat/psyche
npm install

# Install Server dependencies
cd ../server
npm install

# Install WebApp dependencies
cd ../webapp
npm install
```

#### Database Setup
```bash
# Install MySQL (Ubuntu)
sudo apt install mysql-server

# Start MySQL service
sudo systemctl start mysql
sudo systemctl enable mysql

# Create database
mysql -u root -p
```

```sql
CREATE DATABASE navichat;
CREATE USER 'navichat_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON navichat.* TO 'navichat_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### Configure Environment
```bash
# Copy example configuration
cp server/config.example.toml server/config.toml

# Edit configuration file
nano server/config.toml
```

```toml
[database]
host = "localhost"
user = "navichat_user"
password = "your_password"
database = "navichat"

[server]
port = 8080
host = "0.0.0.0"
```

## Verification and Testing

### 1. Test BvhToDeepMimic
```bash
cd BvhToDeepMimic
python example_script.py
```

### 2. Test PyTorch DeepMimic
```bash
cd pytorch_DeepMimic/deepmimic
python testrl.py --arg_file run_humanoid3d_walk_args.txt
```

### 3. Test RSMT
```bash
cd RSMT-Realtime-Stylized-Motion-Transition
python hello.py
```

### 4. Test Chat Interface
```bash
# Build webapp
cd chat/webapp
npm run build

# Start server
cd ../server
node server.js

# In another terminal, start webapp
cd ../webapp
node server.js
```

## Common Installation Issues

### Python Environment Issues

#### Issue: Package conflicts
```bash
# Solution: Use virtual environments
python -m venv motion_env
source motion_env/bin/activate
pip install --upgrade pip setuptools wheel
```

#### Issue: CUDA version mismatch
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
pip install torch==1.12.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### Node.js Issues

#### Issue: Permission errors
```bash
# Use nvm for Node.js version management
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 16
nvm use 16
```

#### Issue: Native dependencies compilation
```bash
# Ubuntu: Install build tools
sudo apt install build-essential

# macOS: Install Xcode command line tools
xcode-select --install
```

### Database Issues

#### Issue: MySQL connection errors
```bash
# Check MySQL status
sudo systemctl status mysql

# Reset MySQL password
sudo mysql_secure_installation
```

#### Issue: Permission denied
```bash
# Fix MySQL socket permissions
sudo chown mysql:mysql /var/run/mysqld/mysqld.sock
```

## Development Environment Setup

### VS Code Extensions
```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-vscode.node-debug2
code --install-extension bradlc.vscode-tailwindcss
code --install-extension ms-vscode.vscode-json
```

### Git Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

### Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export MOTION_ROOT="/path/to/motion"
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

## Performance Optimization

### Python Optimization
```bash
# Install optimized BLAS libraries
sudo apt install libopenblas-dev

# Use conda for better package management
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### GPU Optimization
```bash
# Enable GPU persistence mode
sudo nvidia-smi -pm 1

# Set GPU performance mode
sudo nvidia-smi -ac 5001,1590  # Adjust values for your GPU
```

### System Optimization
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize Python performance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
```

## Docker Installation (Alternative)

### Build Docker Images
```bash
# Build BvhToDeepMimic image
cd BvhToDeepMimic
docker build -t motion/bvh-converter .

# Build PyTorch DeepMimic image
cd ../pytorch_DeepMimic
docker build -t motion/deepmimic .

# Build RSMT image
cd ../RSMT-Realtime-Stylized-Motion-Transition
docker build -t motion/rsmt .

# Build Chat interface
cd ../chat
docker-compose build
```

### Run with Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: navichat
    ports:
      - "3306:3306"
  
  server:
    build: ./chat/server
    ports:
      - "8080:8080"
    depends_on:
      - mysql
  
  webapp:
    build: ./chat/webapp
    ports:
      - "3000:3000"
    depends_on:
      - server
```

## Post-Installation Steps

### 1. Download Assets
```bash
# Download sample BVH files
mkdir -p assets/bvh
wget https://example.com/sample.bvh -O assets/bvh/sample.bvh

# Download 3D assets for chat interface
mkdir -p chat/assets/models
# Add your 3D model files here
```

### 2. Configure Services
```bash
# Set up system services for auto-start
sudo systemctl enable mysql
sudo systemctl enable nginx  # If using nginx as reverse proxy
```

### 3. Security Setup
```bash
# Set up firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 3000  # Development server
sudo ufw enable
```

### 4. Backup Configuration
```bash
# Create backup of configuration files
mkdir -p ~/motion-backup
cp chat/server/config.toml ~/motion-backup/
cp chat/webapp/package.json ~/motion-backup/
```

## Next Steps

1. **Read Project Documentation:** Review individual project docs in the `docs/` folder
2. **Run Examples:** Try the example scripts in each project
3. **Configure Development Environment:** Set up IDE and debugging tools
4. **Join Community:** Connect with other developers and contributors

For project-specific usage instructions, see:
- [BvhToDeepMimic Usage](./BvhToDeepMimic.md#usage)
- [PyTorch DeepMimic Usage](./pytorch_DeepMimic.md#usage)
- [RSMT Usage](./RSMT.md#usage-examples)
- [Chat Interface Usage](./chat_interface.md#usage)
