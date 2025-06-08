#!/bin/bash

echo "Start Server"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

cd src/backend

uvicorn main:app --reload &
BACKEND_PID=$!
cd ..

echo "Start React"
cd frontend
npm start &
FRONTEND_PID=$!
cd ..
cd ..

trap "echo 'Stopping...'; kill -TERM $BACKEND_PID $FRONTEND_PID; wait" SIGINT SIGTERM

wait