#!/bin/bash

echo "Start Server"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "inference for create outputs.json"

python src/backend/inference.py

cd src/backend

uvicorn main:app --reload &
BACKEND_PID=$!
cd ..

echo "Start React"
cd frontend
npm start &
FRONTEND_PID=$!
cd ../..

trap "echo 'Stopping...'; kill $BACKEND_PID -$FRONTEND_PID; wait" SIGINT SIGTERM
wait