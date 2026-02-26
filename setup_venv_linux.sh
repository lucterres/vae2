#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup_venv_linux.sh — Create Python 3.12 venv with PyTorch CUDA 12.1
#
# Usage:
#   bash setup_venv_linux.sh [VENV_PATH]
#
# Default venv location: ~/.venvs/vae2
# Override: bash setup_venv_linux.sh /path/to/my/venv
# -----------------------------------------------------------------------------
set -euo pipefail

VENV_PATH="${1:-.venv}"

# ---------------------------------------------------------------------------
# 1. Locate python3.12
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+')
        if [[ "$version" == "3.12" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.12 not found. Install it first:"
    echo "  sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# ---------------------------------------------------------------------------
# 2. Create venv
# ---------------------------------------------------------------------------
echo ""
echo "Creating venv at: $VENV_PATH"
"$PYTHON" -m venv "$VENV_PATH"

PIP="$VENV_PATH/bin/pip"
PYTHON_VENV="$VENV_PATH/bin/python"

echo "Upgrading pip..."
"$PYTHON_VENV" -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 3. Install PyTorch + Torchvision with CUDA 12.1
# ---------------------------------------------------------------------------
echo ""
echo "Installing torch 2.5.1+cu121 and torchvision 0.20.1+cu121 ..."
"$PIP" install \
    "torch==2.5.1+cu121" \
    "torchvision==0.20.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Install remaining dependencies
# ---------------------------------------------------------------------------
echo ""
echo "Installing remaining dependencies ..."
"$PIP" install \
    pytorch-lightning==2.6.1 \
    numpy==2.3.5 \
    pandas==3.0.0 \
    matplotlib==3.10.8 \
    "opencv-python==4.13.0.92" \
    "pillow==12.1.1" \
    scikit-image

# ---------------------------------------------------------------------------
# 5. Verify CUDA availability
# ---------------------------------------------------------------------------
echo ""
echo "Verifying GPU/CUDA availability ..."
"$PYTHON_VENV" - <<'EOF'
import torch
print(f"torch version      : {torch.__version__}")
print(f"CUDA available     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version       : {torch.version.cuda}")
    print(f"Device count       : {torch.cuda.device_count()}")
    print(f"Device name        : {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not detected. Make sure the NVIDIA driver is installed.")
EOF

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_PATH/bin/activate"
