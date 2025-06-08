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

pip install -r requirements.txt

echo "install node.js dependency..."

cd src/frontend
npm install

echo "Done!"