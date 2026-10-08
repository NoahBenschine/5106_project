# Development Environment Setup Guide (WSL / Linux)

This guide covers setting up the local development environment for the **LLM-Assisted Counterfactual Credit Assignment in MARL** project.

---

## 1. Install Miniconda & Shell Profile Setup

Open your WSL (Ubuntu) terminal and execute the following commands to install Miniconda and configure your shell:

```bash
# Download the latest Linux 64-bit Miniconda installer
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

# Run the installer script
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize Conda for your shell automatically
eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda init

# Reload your shell profile
source ~/.bashrc

# Clean up installer script
rm Miniconda3-latest-Linux-x86_64.sh

# Create dedicated Conda environment
conda create -n marl_project python=3.9 -y

# Activate environment
conda activate marl_project

# Install PyTorch (CPU-only for standard laptop development)
pip install torch torchvision torchaudio

# Install PettingZoo, Gymnasium, Overcooked-AI, and Google GenAI SDK
pip install pettingzoo gymnasium overcooked-ai google-genai

# Clone the repository
git clone [https://github.com/AILabDsUnipi/pymarlzooplus.git](https://github.com/AILabDsUnipi/pymarlzooplus.git)

# Navigate into directory
cd pymarlzooplus

# Install repository requirements
pip install -r requirements.txt

#Sanity Check that installs are successful
python -c "import torch; import pettingzoo; import overcooked_ai_py; import google.genai; print('PyTorch Version:', torch.__version__); print('Overcooked & MARL Environment Setup Complete!')"
