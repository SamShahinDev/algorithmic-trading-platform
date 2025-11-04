#!/bin/bash

echo "🚀 Starting NQ Trading Web Platform"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the backend server
echo ""
echo "✅ Starting FastAPI backend on http://localhost:8000"
echo "📊 Frontend dashboard will be available at http://localhost:8000/frontend/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
cd backend
python3 app.py