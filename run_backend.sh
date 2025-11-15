#!/bin/bash

# Backend Server Startup Script

echo "🚀 Starting Unified Smart Storage Backend..."

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/smart_storage
MONGODB_URL=mongodb://localhost:27017/
MONGODB_DB=smart_storage

# Storage Configuration
STORAGE_BASE_PATH=storage/media

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file with default settings"
    echo "⚠️  Make sure PostgreSQL and MongoDB are running!"
fi

# Start server
echo "🌟 Starting FastAPI server on http://localhost:8000"
echo "📝 API docs available at http://localhost:8000/docs"
echo ""
uvicorn main:app --reload

