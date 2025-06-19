#!/bin/bash

set -e

echo "generate python virtual environment and activate..."

python -m venv venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "generate node.js environment..."

pip install nodeenv
nodeenv -p

echo "install python dependency..."

pip install -r requirements2.txt

read -p "Do you want to install the GPU version of PyTorch? (y/n): " use_gpu

if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip3 install torch torchvision torchaudio
fi

echo "install node.js dependency..."

cd src/frontend
npm install

echo "Done!"