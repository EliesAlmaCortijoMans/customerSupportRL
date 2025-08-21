#!/bin/bash

# Customer Support RL Environment - Quick Start Script

echo "🚀 Customer Support RL Environment - Quick Start"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Install Python dependencies if requirements.txt is newer than last install
if [ requirements.txt -nt .install_marker ] || [ ! -f .install_marker ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch .install_marker
        echo "✅ Python dependencies installed"
    else
        echo "❌ Failed to install Python dependencies"
        exit 1
    fi
fi

# Install Node.js dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd frontend
    npm install
    if [ $? -eq 0 ]; then
        echo "✅ Node.js dependencies installed"
        cd ..
    else
        echo "❌ Failed to install Node.js dependencies"
        exit 1
    fi
else
    echo "✅ Node.js dependencies already installed"
fi

# Check ports
echo "🔍 Checking ports..."
if ! check_port 8000; then
    echo "   Backend port 8000 is busy - please stop the existing server or use a different port"
    exit 1
fi

if ! check_port 3000; then
    echo "   Frontend port 3000 is busy - please stop the existing server or use a different port"
    exit 1
fi

echo "✅ Ports 8000 and 3000 are available"

# Create necessary directories
mkdir -p models logs

echo ""
echo "🎮 Starting Customer Support RL Environment..."
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🔧 Starting backend server (port 8000)..."
python run_server.py serve --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend server failed to start"
    exit 1
fi

echo "✅ Backend server started (PID: $BACKEND_PID)"

# Start frontend server
echo "🎨 Starting frontend server (port 3000)..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 5

echo ""
echo "🎉 System is ready!"
echo ""
echo "📱 Frontend:  http://localhost:3000"
echo "🔧 Backend:   http://localhost:8000"
echo "📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "💡 Demo:      python demo_script.py"
echo "🏋️  Training:  python run_server.py train"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for user interrupt
wait
