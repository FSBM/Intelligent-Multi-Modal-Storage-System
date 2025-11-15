#!/bin/bash

# Frontend Server Startup Script

echo "🚀 Starting Unified Smart Storage Frontend..."

# Navigate to frontend directory
cd "$(dirname "$0")/frontend" || exit

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📥 Installing dependencies (this may take a few minutes)..."
    npm install
fi

# Start server
echo "🌟 Starting Next.js server on http://localhost:3000"
echo ""
npm run dev

